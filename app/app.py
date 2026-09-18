"""
SIT307 8.1D - Sydney Housing Price Decision Support System
Streamlit front end for the pipeline trained in SIT307-8.1D-analysis.ipynb.

Run from the project root:
    streamlit run app/app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

st.set_page_config(page_title="Sydney Housing Price Estimator", page_icon="🏠",
                   layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load(APP_DIR / "model.joblib")
    meta = json.loads((APP_DIR / "model_meta.json").read_text())
    return model, meta


@st.cache_data
def load_sales():
    return pd.read_csv(ROOT / "data" / "sydney_housing_clean.csv", parse_dates=["sold_date"])


model, meta = load_model()
sales = load_sales()


def build_features(suburb, property_type, beds, baths, parking, land_size):
    """Reproduce exactly the feature engineering used during training.

    land_size is None when the figure is unknown, which is a real and common case:
    the Domain search cards omit land size for many houses, including the property
    that produced the single largest error in testing.
    """
    land = float(land_size) if (property_type == "House" and land_size) else np.nan
    known = not np.isnan(land)
    row = {
        "beds": beds,
        "baths": baths,
        "parking": parking,
        "total_rooms": beds + baths,
        "bath_per_bed": baths / max(beds, 1),
        "log_land_size": np.log1p(land if known else 0.0),
        "land_per_bed": (land if known else 0.0) / max(beds, 1),
        "has_parking": int(parking > 0),
        "has_land_data": int(known),
    }
    row["suburb"] = suburb
    row["property_type"] = property_type
    return pd.DataFrame([row])[meta["numeric_features"] + meta["categorical_features"]]


def comparables(suburb, property_type, beds, k=3):
    """Closest matches in the training data, ranked by suburb, then type, then bedrooms."""
    pool = sales.copy()
    pool["score"] = (
        (pool.suburb != suburb) * 100
        + (pool.property_type != property_type) * 10
        + (pool.beds - beds).abs()
    )
    return pool.sort_values(["score", "sold_date"], ascending=[True, False]).head(k)


# ----------------------------------------------------------------- sidebar
st.sidebar.header("Property details")
suburb = st.sidebar.selectbox("Suburb", meta["suburbs"], index=1)
property_type = st.sidebar.selectbox("Property type", meta["property_types"])
beds = st.sidebar.slider("Bedrooms", 0, 7, 3)
baths = st.sidebar.slider("Bathrooms", 1, 5, 2)
parking = st.sidebar.slider("Parking spaces", 0, 6, 1)

land_size = None
if property_type == "House":
    land_known = st.sidebar.checkbox("Land size known", value=True)
    if land_known:
        land_size = st.sidebar.number_input("Land size (m²)", min_value=100, max_value=2000,
                                            value=560, step=10)
    else:
        st.sidebar.caption("Modelled as a house with no land figure — the same condition that "
                           "produced the largest error in testing. Expect a wider true range "
                           "than the band below suggests.")
else:
    st.sidebar.caption("Land size is only used for houses. For units and townhouses the listing "
                       "figure is the whole strata block, not the dwelling.")

st.sidebar.divider()
st.sidebar.caption(
    f"Model: {meta['model']} trained on {meta['trained_on']} sales collected 17 Sep 2026 "
    f"from Domain."
)

# ----------------------------------------------------------------- main
st.title("🏠 Sydney Housing Price Decision Support System")
st.caption("SIT307 8.1D — Saatvik Sharma (225158822) · Blacktown · Parramatta · Mosman")

X = build_features(suburb, property_type, beds, baths, parking, land_size)
predicted = float(np.exp(model.predict(X)[0]))

band = meta["cv_mape"] / 100
low, high = predicted * (1 - band), predicted * (1 + band)

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Estimated sale price")
    st.metric(label=f"{beds} bed · {baths} bath · {parking} car {property_type.lower()} "
                    f"in {suburb}",
              value=f"${predicted:,.0f}")
    # NOTE: escape the dollar signs - Streamlit renders $...$ as LaTeX otherwise.
    st.write(f"**Likely range:** \\${low:,.0f} — \\${high:,.0f}  "
             f"(±{meta['cv_mape']:.1f}%, which contained the true price for "
             f"{meta['band_coverage_pct']:.0f}% of properties in testing)")
    st.progress(min(1.0, predicted / 6_000_000))

    if predicted > meta["caution_threshold"]:
        st.warning(
            "**Treat this estimate with caution.** Above roughly \\$3m the model has few "
            "comparable sales to learn from and its errors are large in both directions — all "
            "five of the worst predictions in testing were Mosman houses in this range, missing "
            "by between −49% and +109%. Price here is set by outlook, aspect and street position, "
            "none of which this model can see."
        )
    elif suburb == "Mosman" and property_type == "House":
        st.info(
            "Mosman houses vary enormously by position within the suburb. Use this as a starting "
            "point for a comparable-sales analysis, not as a valuation."
        )
    else:
        st.success(
            "This property sits in a well-represented segment of the training data, where the "
            "model is most reliable."
        )

with right:
    st.subheader("Model performance")
    st.dataframe(
        pd.DataFrame({
            "Metric": ["Cross-validated MAE", "Mean absolute % error",
                       "Median absolute % error", "R² on log(price)", "Training sales"],
            "Value": [f"${meta['cv_mae']:,.0f}",
                      f"{meta['cv_mape']:.1f}% ± {meta['cv_mape_sd']:.1f}",
                      f"{meta['median_abs_pct_error']:.1f}%",
                      f"{meta['cv_r2_log']:.3f}",
                      f"{meta['trained_on']}"],
        }), width="stretch", hide_index=True,
    )
    st.caption("Averaged over five different cross-validation splits, so every figure comes from "
               "predictions on properties the model had not seen. The ± is the spread across "
               "those splits.")

st.divider()
st.subheader("Comparable recent sales")
comps = comparables(suburb, property_type, beds)
st.dataframe(
    comps[["address", "suburb", "property_type", "beds", "baths", "parking",
           "land_size_m2", "sold_price", "sold_date"]]
    .rename(columns={"land_size_m2": "land (m²)", "sold_price": "sold for",
                     "sold_date": "sold on"})
    .style.format({"sold for": "${:,.0f}", "land (m²)": "{:,.0f}",
                   "sold on": lambda d: d.strftime("%d %b %Y")}, na_rep="—"),
    width="stretch", hide_index=True,
)

with st.expander("How this estimate is produced, and what it cannot do"):
    st.markdown(f"""
A **{meta['model']}** fitted to `log(sale price)` on {meta['trained_on']} sales hand-collected
from Domain across Blacktown, Parramatta and Mosman, with predictions converted back to dollars.
The quoted range is ±{meta['cv_mape']:.1f}%, the model's mean absolute percentage error across
five cross-validation splits. That band is a typical-error width, not a confidence interval: it
contained the true sale price for {meta['band_coverage_pct']:.0f}% of properties tested.

**It cannot see** internal floor area, street position within the suburb, outlook or aspect,
renovation quality, floor level, strata levies, planning overlays or zoning. Those omissions cause
every large error observed in testing.

**It deliberately does not ask** whether the property sold at auction. That is known only at or
after the sale, so a tool valuing a property beforehand cannot use it — and in testing it earned
nothing anyway.

This is a decision support input for a comparable-sales analysis. It is not a valuation and should
not be relied on for a financial decision.
""")
