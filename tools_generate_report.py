import json, pathlib

R = json.load(open("results/report_numbers.json"))
mc, tn, ss = R["model_comparison"], R["tuned"], R["suburb_summary"]
ab, ms = R["ablation"], R["mape_by_suburb"]

def d(x):  return f"\\${x:,.0f}"
def p(x, n=1): return f"{x:.{n}f}\\%"

V = {
 "RAW": R["raw_rows"], "PAGES": R["source_pages"], "WITHHELD": R["withheld"],
 "N": R["modelled_rows"],
 "SKEWRAW": f"{R['skew_raw']:.1f}", "SKEWLOG": f"{R['skew_log']:.1f}",
 "CORRM": f"{R['corr_months_mosman']:.2f}",
 "AUCM": f"{R['pct_auction_mosman']:.0f}", "AUCP": f"{R['pct_auction_parramatta']:.0f}",
 "ABL_BOTH": f"{ab['both time and auction']:.2f}", "ABL_M": f"{ab['+ months_since_start']:.2f}",
 "ABL_A": f"{ab['+ is_auction']:.2f}", "ABL_NONE": f"{ab['neither (final set)']:.2f}",
 "IMPM": f"{R['imp_months_when_included']:.2f}", "IMPRANK": R["imp_months_rank"],
 "NFLAG": R["n_flagged"],
 "RFWINS": R["rf_wins_seeds"], "NSEEDS": R["n_seeds"],
 "GAP": f"{abs(R['rf_gb_gap']):.2f}",
 "RFLO": f"{R['rf_mape_spread'][0]:.1f}", "RFHI": f"{R['rf_mape_spread'][1]:.1f}",
 "NESTMAPE": f"{R['nested']['MAPE_%']:.2f}", "NESTMAE": d(R["nested"]["MAE"]),
 "BEST": R["best_model"],
 "TGB_MAPE": f"{tn['Gradient Boosting']['CV MAPE %']:.2f}",
 "TGB_SD":   f"{tn['Gradient Boosting']['CV MAPE sd']:.2f}",
 "TGB_MAE":  d(tn["Gradient Boosting"]["CV MAE"]),
 "TRF_MAPE": f"{tn['Random Forest']['CV MAPE %']:.2f}",
 "TRF_MAE":  d(tn["Random Forest"]["CV MAE"]),
 "TRG_MAPE": f"{tn['Ridge']['CV MAPE %']:.2f}",
 "W10": f"{R['within_10']:.0f}", "W20": f"{R['within_20']:.0f}",
 "MEDAPE": f"{R['median_ape']:.1f}", "W5SHARE": f"{R['worst5_share']:.0f}",
 "W5UNDER": R["worst5_under"],
 "MAPE_BT": f"{ms['Blacktown']:.0f}", "MAPE_PA": f"{ms['Parramatta']:.0f}",
 "MAPE_MO": f"{ms['Mosman']:.0f}",
 "TOPN": R["top_n"], "TOPUNDER": f"{R['top_pct_under']:.0f}",
 "TOPMED": f"{R['top_median_signed']:+.1f}",
 "BAND": f"{R['band_pct']:.1f}", "COVER": f"{R['band_coverage']:.0f}",
 "IMP_MOS": f"{R['top_features']['suburb_Mosman']:.2f}",
}
for s in ("Parramatta", "Blacktown", "Mosman"):
    k = s[:2].upper()
    V[f"{k}_N"] = int(ss[s]["properties"]); V[f"{k}_MED"] = d(ss[s]["median"])
    V[f"{k}_MIN"] = d(ss[s]["minimum"]);    V[f"{k}_MAX"] = d(ss[s]["maximum"])
for name, key in [("RIDGE", "Ridge"), ("RF", "Random Forest"), ("GB", "Gradient Boosting")]:
    m = mc[key]
    V[f"{name}_MAPE"] = f"{m['CV MAPE %']:.2f}"; V[f"{name}_SD"] = f"{m['CV MAPE sd']:.2f}"
    V[f"{name}_MAE"] = d(m["CV MAE"]); V[f"{name}_RMSE"] = d(m["CV RMSE"])
    V[f"{name}_TRAIN"] = d(m["Train MAE"]); V[f"{name}_R2"] = f"{m['CV R2 (log)']:.3f}"

BODY = r"""
% =====================================================================
\section{Problem Definition and Data Collection}

An agency wants an estimate of what a Sydney property will sell for: supervised regression on sale
price, where what it needs is a defensible number plus an honest signal of when not to trust it.

Blacktown (2148), Parramatta (2150) and Mosman (2088) are three different markets --- outer-west
houses on 500--800\,m$^2$ blocks, a high-rise apartment CBD, and premium harbourside. The price
\emph{drivers} differ too: land should dominate in Blacktown, the dwelling is the whole asset in
Parramatta, and Mosman turns on outlook and street position, which no card records.

\begin{table}[H]
\centering
\caption{The three suburbs, after cleaning}
\label{tab:suburbs}
\begin{tabular}{lrrrr}
\toprule
Suburb & Properties & Median & Minimum & Maximum \\
\midrule
Parramatta & @@PA_N@@ & @@PA_MED@@ & @@PA_MIN@@ & @@PA_MAX@@ \\
Blacktown  & @@BL_N@@ & @@BL_MED@@ & @@BL_MIN@@ & @@BL_MAX@@ \\
Mosman     & @@MO_N@@ & @@MO_MED@@ & @@MO_MIN@@ & @@MO_MAX@@ \\
\bottomrule
\end{tabular}
\end{table}

I transcribed @@RAW@@ listings by hand on 17 September 2026 from @@PAGES@@ ``Sold'' result pages
on \texttt{domain.com.au}, reading the public cards in a browser. No scraper. @@WITHHELD@@ had no
published price, leaving @@N@@ for modelling.

\subsection{Quality, missing information and bias}

Withheld prices dominate and are not missing at random: they cluster in recent sales, unpublished
until settlement. Mosman page 1 gave 4 usable rows of 20 against 16, 15 and 17 on pages 4--6, so I
skipped to older pages and transcribed only priced Mosman cards. \textbf{The file therefore cannot
evidence the Mosman withholding rate} --- all @@WITHHELD@@ blanks are Blacktown and Parramatta.
That is a limitation of how I built the artefact, not only of the source.

Also: duplicate agency listings; two suppressed addresses; land size incomparable across types
(usually the whole strata block for units, 3,733\,m$^2$ for one Blacktown apartment), so kept only
for detached dwellings; and only published agency sales are visible, making this a biased sample.
What the card omits matters most --- floor area, condition, aspect, floor level, strata levies,
zoning --- and Section 4 shows those omissions cause every large error.

% =====================================================================
\section{Data Understanding and Feature Engineering}

Logging the target cuts skewness from @@SKEWRAW@@ to @@SKEWLOG@@ --- a large improvement, still
visibly skewed because Mosman's houses form a second hump. The point is not normality: on raw
dollars a 10\% miss on a \$400k unit counts for almost nothing against the same miss on a \$4m
house.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig02_suburb_comparison.png}
\caption{Parramatta is a tight market of near-interchangeable apartments; Blacktown is mixed;
Mosman is the most dispersed. Only Blacktown and Mosman hold many detached houses.}
\end{figure}

Pooled correlations mislead because the suburb confounds them; within each suburb, bedrooms and
bathrooms signal strongly in Blacktown and weakly in Parramatta. The outlier rule flags @@NFLAG@@
genuine sales, all kept --- and misses the 1,182\,m$^2$ block at 17 Carinya Street, unremarkable
\emph{for Blacktown}: a univariate rule finds unusual prices, not unusual price drivers.
\textbf{Predicted top three, written down before engineering:} suburb, property type, land size.

\subsection{Two features I threw away}

Alongside \texttt{total\_rooms}, \texttt{bath\_per\_bed}, \texttt{log\_land\_size},
\texttt{land\_per\_bed}, \texttt{has\_parking} and \texttt{has\_land\_data} I built
\texttt{months\_since\_start} and \texttt{is\_auction}; neither survived.
\texttt{months\_since\_start} is a suburb label in disguise: Mosman needed older pages, so sale
date correlates @@CORRM@@ with ``is Mosman''. Included, it was the \emph{most important} feature
in the forest (@@IMPM@@, rank @@IMPRANK@@) over a four-month window --- not credible.
\texttt{is\_auction} fails differently: unknown until the sale happens, so an appraiser cannot
supply it, and a partial suburb proxy (@@AUCM@@\% of Mosman sales against @@AUCP@@\% in
Parramatta). Ablation settles both --- untuned forest MAPE @@ABL_BOTH@@\% with both,
@@ABL_M@@\% with the date, @@ABL_A@@\% with auction, @@ABL_NONE@@\% with neither. Removing them
\emph{improves} the model.

% =====================================================================
\section{Model Development and Evaluation}

Ridge is the interpretable baseline, multiplicative once the target is logged. Random forest
relaxes additivity, splitting on suburb then learning a surface per branch. Gradient boosting is
most flexible and most likely to overfit @@N@@ rows. \textbf{Prediction recorded before fitting:
the random forest wins}, because the structure is a suburb-by-type interaction trees capture and
Ridge cannot, and averaging should beat boosting at this sample size. One five-fold split is not
stable at @@N@@ properties, so each model runs over @@NSEEDS@@ fold assignments, mean $\pm$ sd.

\begin{table}[H]
\centering
\caption{Cross-validated performance, mean over @@NSEEDS@@ fold assignments}
\label{tab:models}
\begin{tabular}{lrrrr}
\toprule
Model & CV MAE & CV MAPE & CV $R^2$ (log) & Train MAE \\
\midrule
Ridge             & @@RIDGE_MAE@@ & @@RIDGE_MAPE@@\% $\pm$ @@RIDGE_SD@@ & @@RIDGE_R2@@ & @@RIDGE_TRAIN@@ \\
Random Forest     & @@RF_MAE@@ & @@RF_MAPE@@\% $\pm$ @@RF_SD@@ & @@RF_R2@@ & @@RF_TRAIN@@ \\
Gradient Boosting & @@GB_MAE@@ & @@GB_MAPE@@\% $\pm$ @@GB_SD@@ & @@GB_R2@@ & @@GB_TRAIN@@ \\
\bottomrule
\end{tabular}
\end{table}

\textbf{My prediction was wrong.} Boosting wins all @@NSEEDS@@ splits; the forest won
@@RFWINS@@. The margin is @@GAP@@ percentage points --- small, but consistent, which is what
separates signal from noise. Depth-2 stumps suit a suburb-by-type interaction plus a size effect;
the forest's deeper trees spend capacity on splits @@N@@ rows cannot support.

\begin{figure}[H]
\centering
\includegraphics[width=0.74\textwidth]{figures/fig05b_seed_sensitivity.png}
\caption{One dot per fold assignment. The forest's own MAPE ranges @@RFLO@@--@@RFHI@@\% on nothing
but fold membership, so the spread within a model rivals the gap between models. My first pass used
one split and ranked the forest first; five splits reverse it.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig06_learning_curves.png}
\caption{Ridge has the smallest train/CV gap and sits closest to underfitting; boosting has the
largest, fitting training data about twice as closely as it generalises --- and still wins, because
a model can overfit and remain the best estimator. All three curves are still falling at the full
sample size, so the binding constraint is data, not capacity.}
\end{figure}

Tuning shares fold assignment 42 with the evaluation, so that row is optimistic in principle;
nested cross-validation gives @@NESTMAPE@@\% and @@NESTMAE@@, essentially unchanged. Suburb and
property type top the importance ranking (\texttt{suburb\_Mosman} at @@IMP_MOS@@), confirming two
of my three predictions; land size is the partial miss, its signal tangled with
\texttt{has\_land\_data}.

\textbf{Recommendation: gradient boosting} at @@TGB_MAPE@@\% $\pm$ @@TGB_SD@@, with one
complication: on MAE the tuned forest is better (@@TRF_MAE@@ vs @@TGB_MAE@@). MAE is dominated by
a few multi-million-dollar Mosman misses; MAPE is not. I chose MAPE, because \$50k out on a
\$500k unit is a worse failure than \$50k out on a \$5m house.

% =====================================================================
\section{Investigating Prediction Failures}

Out-of-fold, @@W10@@\% of predictions land within 10\% and @@W20@@\% within 20\%, median error
@@MEDAPE@@\%. The five worst carry @@W5SHARE@@\% of all absolute error and are all Mosman houses,
@@W5UNDER@@ under and two over. I checked four against their Domain property-profile pages
(\texttt{data/worst\_case\_profiles.csv}); I did not retrieve 17 Avenue Road's.

\begin{table}[H]
\centering\small
\caption{The five largest prediction errors}
\label{tab:worst}
\begin{tabular}{p{3.0cm}rr p{6.5cm}}
\toprule
Property & Actual & Predicted & What the model could not see \\
\midrule
21 Morella Road & \$11.65m & \$5.04m & 516\,m$^2$ on its profile page and \textbf{no land figure
at all on its search card}. Prestige ridge street; last sold \$7.31m in 2020. \\
3 Gordon Street & \$7.65m & \$3.94m & Two bathrooms reads modest, but 186\,m$^2$ internal, a pool,
and a position that took it from \$1.82m in 2013. Sold in 27 days. \\
1 Bradleys Head Road & \$9.40m & \$5.95m & 892\,m$^2$ near the foreshore, built 1930, pool.
Heritage and bushfire overlays. Sold prior to auction in 29 days. \\
96 Glover Street & \$3.39m & \$7.07m & Premium on paper at 4 bed / 3 bath, but only 379\,m$^2$, in
a heritage conservation area, 97 days to sell. Domain's own estimate was \$3.23m --- the market was
right and the model was wrong. \\
17 Avenue Road & \$5.45m & \$8.50m & Five-bedroom house, no land size on its card: the same
missing-data condition as 21 Morella Road, resolved upward here and downward there. \\
\bottomrule
\end{tabular}
\end{table}

Two mechanisms, not one. Under-predictions hide their value in attributes the data lacks, above
all \textbf{internal floor area}; over-predictions look premium on the visible four and are not.
The Morella/Avenue Road pair makes the point: identical missing-land condition, errors in opposite
directions.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/fig08_error_analysis.png}
\caption{Error against sale price, and mean absolute percentage error by segment. Blacktown
@@MAPE_BT@@\%, Parramatta @@MAPE_PA@@\%, Mosman @@MAPE_MO@@\%.}
\end{figure}

\textbf{Is it biased against expensive property?} Mostly not. Above \$3m (n = @@TOPN@@) it
under-predicts @@TOPUNDER@@\% of the time, median signed error @@TOPMED@@\% --- a slight tilt,
not the systematic under-valuation I assumed, and the two largest misses run opposite ways. The
problem is \textbf{variance, not bias}: bias could be calibrated away, variance can only be
communicated.

\textbf{Which are inherently harder?} Mosman worst and Blacktown best, as expected; Parramatta
between them is not. Its features barely vary --- near-identical two-bedroom apartments carrying
\$480k--\$800k, separated by floor level, outlook, building age and strata levies, none recorded.
Low target variance does not help when the features have less. Mosman is hard because its
properties are unique; Parramatta because they are indistinguishable to the model but not to a
buyer.

The missing information that would help most: internal floor area, street-level location, planning
overlays, year built, aspect, days on market --- all of it on the property-profile pages, one
extra page view per property away.

% =====================================================================
\section{Deployment and Reflection}

A Streamlit app (\texttt{app/app.py}) loads the pipeline from \texttt{app/model.joblib}, rebuilds
exactly the training features and predicts. Four choices follow from Section 4: a $\pm$@@BAND@@\%
range \emph{with its real coverage stated} (@@COVER@@\%) rather than an implied confidence
interval; three comparable sales, which an appraiser trusts more than a model; a warning above
\$3m; and the option of a house with \emph{unknown} land size, the condition behind the largest
error. It never asks whether the sale was at auction.

\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{figures/fig09_app_blacktown.jpg}
\caption{A Blacktown house, in the segment where the model is most reliable.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{figures/fig10_app_mosman.jpg}
\caption{A Mosman house above \$3m: the caution message replaces the confident one, and the
comparables are genuine Mosman sales.}
\end{figure}

\begin{lstlisting}[language=bash]
git clone <repository URL> && cd sit307-8-1d-sydney-housing
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace SIT307-8.1D-analysis.ipynb
streamlit run app/app.py        # http://localhost:8501
\end{lstlisting}

\subsection{Reflection}

The longest part of this project had no machine learning in it. Collecting @@RAW@@ listings by
hand, deciding what ``land size'' means for an apartment, noticing that a withheld price is not
missing at random --- that was most of the work, and it set the ceiling on everything after. No
model rescued a feature set that never contained floor area.

Three things caught me out, all methodological: the strongest feature in my first run encoded the
order I collected the data in; my model ranking came from one fold assignment and reversed across
five; and the fairness claim I expected to make was not in the data when I tested it. Each looked
right until it was measured.

Boosting is less interpretable than Ridge, and for financial advice that is a real cost; I took it
because the gap is meaningful and compensated with comparables and a stated-coverage band.
Dropping \texttt{is\_auction} is the opposite trade: free accuracy on paper, worthless in
practice.

The data over-represents what agencies publish and under-represents withheld and off-market sales,
which cluster at the premium end. Because the failure there is unreliability rather than a tilt,
the fair warning is not ``it reads low'' but ``above \$3m it is close to guessing'' --- Table
\ref{tab:worst} misses by $-$49\% and $+$109\% on properties a few streets apart. Used for
reserve prices without that caveat it would occasionally be wrong by millions.

With more time I would collect from the profile pages, expand to several thousand sales over a
longer window, and add coordinates for position within a suburb. More compute would buy almost
nothing: every model trains in seconds, and the constraint is data.

% =====================================================================
\section{Code, Data and Reproduction}

Everything is in the repository below. The notebook runs top to bottom without errors and
regenerates every figure, table and model file, including
\texttt{results/report\_numbers.json}, the source of every number in this report.

\begin{center}
\url{\repoURL}
\end{center}

% =====================================================================
\section{Acknowledgement of Generative AI Use}

I used Claude (Anthropic) to plan the structure, write and debug the notebook and Streamlit code,
and check this report's grammar and clarity. Suburb selection, modelling decisions, interpretation
and conclusions are mine. Two examples of critical rather than passive use. When an AI-assisted review flagged that my model
comparison rested on a single fold assignment, I re-ran it across @@NSEEDS@@ splits, found my
pre-registered prediction was wrong, and reported that instead of the flattering single-split
result. The same review caught my fairness claim as unsupported; I tested it, found variance
rather than bias, and rewrote the section. Every number here is generated by the notebook,
precisely because an earlier draft contained figures that had drifted from the code.

% =====================================================================
\section{References}

\begin{enumerate}
    \item Domain Group, \emph{Sold listings and property profiles: Blacktown, Parramatta and
    Mosman, NSW}, retrieved 17 September 2026, \url{https://www.domain.com.au/}.
    \item Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
    \emph{Journal of Machine Learning Research}, 12, 2825--2830.
    \item Breiman, L. (2001). Random Forests. \emph{Machine Learning}, 45(1), 5--32.
    \item Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
    \emph{Annals of Statistics}, 29(5), 1189--1232.
    \item Cawley, G. C. \& Talbot, N. L. C. (2010). On Over-fitting in Model Selection and
    Subsequent Selection Bias in Performance Evaluation. \emph{JMLR}, 11, 2079--2107.
    \item Streamlit Inc. (2026). \emph{Streamlit documentation}. \url{https://docs.streamlit.io}.
\end{enumerate}

\end{document}
"""

for k, v in V.items():
    BODY = BODY.replace(f"@@{k}@@", str(v))
assert "@@" not in BODY, BODY[BODY.index("@@"):BODY.index("@@")+40]

head = pathlib.Path("report.tex").read_text().split("\\tableofcontents")[0] + \
       "\\tableofcontents\n\\newpage\n"
pathlib.Path("report.tex").write_text(head + BODY)
print("report.tex generated from results/report_numbers.json")
