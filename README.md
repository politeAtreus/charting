# 📈 CSV Plotter

Upload a CSV, pick columns, and plot them. Handles messy files, dual Y-axes, and optional power/energy calculations.

---

## What it does

- **Plots any CSV** - drag in a file and start charting. Delimiter and header row are detected automatically
- **Dual Y-axes** - put different columns on left and right Y axes so mismatched scales don't ruin the chart
- **Column scaling** - multiply any column by a number before plotting. Useful for unit conversion (e.g. `current_A × 1000` to plot in mA)
- **Power & energy** - if you have voltage and current columns, it can compute power (W) and cumulative energy (Wh) using trapezoidal integration
- **Stage filtering** - if your CSV has a `Status` column, filter rows by status to focus on a specific test phase
- **Two-file compare** - upload two CSVs at once and plot them together. Columns are prefixed `1_` and `2_` to tell them apart
- **Export** - download the cleaned CSV or a numeric summary

---

## Setup

```bash
git clone https://github.com/politeAtreus/charting/
cd charting
pip install -r requirements.txt
streamlit run charting.py
```

Then open `http://localhost:8501`.

---

## How to use

1. Upload a CSV file (or two) at the top of the page
2. Adjust any parse settings in the sidebar if needed
3. Scroll to **Chart**, pick your X axis and Y columns, and plot
4. Use **Downloads** at the bottom to export

### Sidebar settings

| Setting | What it does |
|---------|-------------|
| Header row | `auto` finds the header automatically. Set a number if it's on a specific row |
| Delimiter | Auto-detected, but you can force `,` `;` `\t` or `\|` |
| Interpret % as fractions | Turns `45%` into `0.45` |
| Compute Power/Energy | Adds power (W) and energy (Wh) columns computed from your voltage and current columns |
| Column Scaling | Pick a column and a multiplier. Applied before plotting |
| Log axis | Toggle log scale on X, left Y, or right Y independently |

---

## Handling messy CSVs

The parser tries to do the right thing with imperfect files:

- **Metadata above the header** - rows before the header are shown in a collapsible section, not mixed into the data
- **Sparse columns** - if a column is only filled in for some rows, it's still treated as numeric. The numeric check runs on populated rows only, not the whole column
- **Mixed types** - booleans (`true`/`false` → `1`/`0`), percentages, currency symbols, and unit suffixes (e.g. `12V`) are stripped and converted automatically
- **String columns with few values** - label-encoded to integers so they can be plotted

---

## Requirements

See [`requirements.txt`](requirements.txt). Main dependencies: `streamlit`, `plotly`, `pandas`, `numpy`.

---

## Troubleshooting

**Chart doesn't update after I change the code** - Streamlit caches the parsed data. Clear it via the menu in the top right corner -> *Clear cache*, or run `streamlit cache clear` in the terminal.

**Two files have different row counts** - rows are matched by position (row 1 with row 1, row 2 with row 2), not by time or any key column. The shorter file leaves the extra rows empty.

**Power/Energy result looks wrong** - check that your voltage and current columns are in the same unit system before enabling the calculation. Use column scaling to convert units first if needed.
