# Data collection log — SIT307 8.1D

Collected 17 September 2026 by hand from Domain sold-listing search pages, read through
the browser. No automated scraping was used; each row was transcribed from the visible
result card.

Search URLs (sorted by sale date, newest first, default filters):
- https://www.domain.com.au/sold-listings/blacktown-nsw-2148/ (pages 1-3)
- https://www.domain.com.au/sold-listings/parramatta-nsw-2150/ (pages 1-4)
- https://www.domain.com.au/sold-listings/mosman-nsw-2088/ (pages 1, 4, 5, 6)

## Issues observed during collection

1. **Price withheld.** A material share of cards show "Price Withheld" instead of a sale
   price. Recorded with a blank `sold_price` and dropped before modelling. Blacktown 4/60;
   Parramatta withholds heavily on new high-rise stock; only 4 of the first 20 Mosman results
   published a price, which is why Mosman required pages 4-6 to reach thirty usable sales.
2. **Duplicate cards.** The same property is sometimes listed twice on one page by two
   agencies (e.g. 1001/83 Church Street, Parramatta). Deduplicated on
   address + sold_date + price.
3. **Address suppressed.** Two Blacktown apartments displayed only "Blacktown" with no
   street address; recorded as "Address withheld A/B".
4. **Land size missing.** Absent for most apartments and many townhouses. Where it is
   present for an apartment it is often the whole strata block (e.g. 3,733 m² for a unit
   at 2-4 Fourth Avenue, Blacktown), so apartment land sizes are not comparable to house
   land sizes.
5. **Parking shown as a dash.** Some inner-city apartments display "-" for parking; recorded
   as 0 spaces.
6. **"and surroundings".** The result header reads "... and surroundings", so listings from
   neighbouring suburbs can appear. Only cards whose address ends in the target suburb were
   transcribed.
7. **Survivorship / selection bias.** Only sales the portal chose to publish are visible.
   Private sales, withheld prices and off-market transactions are absent, which biases the
   sample toward agency-marketed stock.

## Totals

164 listings transcribed; 154 with a published sale price form the modelling set
(Blacktown 56, Mosman 52, Parramatta 46).
