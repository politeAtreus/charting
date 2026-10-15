# CSV Plotter

Streamlit and Plotly charting for up to eight hydrophone CSV logs, with independent time alignment, stage filtering, dual Y axes, scaling, and optional power/energy calculations.

## Run

```bash
pip install -r requirements.txt
streamlit run charting.py
```

Open `http://localhost:8501`. If the project is already installed, replace `charting.py` and restart Streamlit; no new application dependencies are needed.

## Compare hydrophones

1. Upload each hydrophone CSV in its own upload slot. Columns are prefixed with the serial number extracted from the filename. Files for the same serial receive distinct suffixes.
2. In the sidebar under **Hydrophone comparison**, select the stages in **Filter by Status**. The filter uses each file's own `Status` column. Clearing every selection hides all stages. A file without `Status` remains unfiltered and is identified in the sidebar.
3. Choose **Align time to**:

   | Option | Behavior |
   | --- | --- |
   | Start of selected data | The first valid time in each hydrophone's filtered data becomes zero. This is the default. |
   | Recorded time | Preserve the elapsed time recorded in each file. |
   | Start of a soak stage | The first valid occurrence of **Stage at time zero** becomes zero for each hydrophone. The anchor can be hidden by the stage filter. A file missing that anchor is omitted with a message. |

4. Expand **Manual time offsets (hours)** to fine-tune any curve. Negative values move it earlier/left; positive values move it later/right. Offsets are added after automatic alignment. **Reset time offsets** sets every manual offset to zero.
5. In **Chart**, choose a `Time(h)` or `Time(s)` X axis, then the desired left and right Y columns. The default X axis is the first hydrophone's `Time(h)` column. Every trace uses the matching time column from its own hydrophone, regardless of which hydrophone's time column is selected.

For example, deselect **Soak Init** and keep **Start of selected data**. Each hydrophone's initialization data is removed independently, and its first remaining sample starts at zero. To move hydrophone 7570 another 30 minutes left, set its offset to `-0.5` hours.

Both `Time(s)` and `Time(h)` are available when either is supplied, even if power/energy calculations are disabled. The original time gaps and differences in sampling intervals are preserved. Aligning one stage shifts the entire curve by a constant; it does not stretch cycles or realign every repeated stage.

Time alignment applies to `Time(s)`/`Time(h)`. Other X columns and `(index)` compare rows by position. A trace without its own matching elapsed-time column is omitted when a time X axis is selected, with a message. It never borrows a different hydrophone's clock.

## Invalid battery data

Rows whose `battStatus` is **No Data Available** are removed before numeric conversion, derived calculations, summaries, and exports. Matching ignores letter case and surrounding spaces. The app reports how many rows were skipped for each file.

This removes startup/restart placeholders such as battery percentage `255` and voltage `0` without discarding genuine zero readings in valid rows. The rule uses the status field, not a numeric threshold. CSVs without `battStatus` keep their original rows.

Skipping a row does not compress the elapsed time. Line charts join the remaining samples across an outage; they do not reconstruct missing readings. Energy integration skips intervals crossing discarded rows rather than estimating energy during the outage.

## Other controls

| Setting | Behavior |
| --- | --- |
| Header row | Detect automatically, treat as no header, or choose a one-based header row. |
| Delimiter | Detect automatically or select comma, semicolon, tab, or pipe. |
| Interpret % as fractions | Convert `45%` to `0.45`. |
| Compute Power/Energy | Use battery mV/mA or DMM V/A with elapsed seconds to calculate W and Wh. |
| Column Scaling | Multiply selected numeric columns before plotting and exporting. |
| Log modes | Select logarithmic X, left Y, and right Y independently; nonpositive values are hidden. |

Metadata before the header is shown separately. The parser handles sparse numeric columns, boolean values, currency symbols, and unit suffixes. `Status` and `battStatus` remain text.

**Downloads** exports the filtered, aligned, scaled comparison table and summaries. Each hydrophone's time columns accompany its data. Files can have different row counts; blank padding in the combined table is excluded from numeric statistics. Source CSV files are never modified.

## Verification

```bash
pip install pytest
python -m pytest -q
```

The regression checks use synthetic CSV logs and Streamlit's AppTest. They cover invalid readings, valid zero readings, per-hydrophone stage filtering, unequal time grids, both time units and Y axes, offsets/reset, hidden/missing anchors, and empty stage selections.

If a running app retains old parsed data after replacing the code, use Streamlit's **Clear cache** menu or `streamlit cache clear`, then restart it.
