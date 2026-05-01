# Primetrade.ai — Data Science Intern Assignment
## Trader Performance vs Market Sentiment (Fear/Greed Index)

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter nbformat
```

Place both CSV files in the same folder:
- `fear_greed_index.csv`
- `historical_data.csv`

Then open the notebook:

```bash
jupyter notebook primetrade_analysis.ipynb
```

---

## Dataset Summary

| Dataset | Rows | Columns | Period |
|---------|------|---------|--------|
| Fear/Greed Index | 2,644 | 4 | Feb 2018 – May 2025 |
| Hyperliquid Trader Data | 211,224 | 16 | May 2023 – May 2025 |

No missing values. No duplicates. Overlapping window: **May 2023 – May 2025 (731 days).**

---

## Methodology

**Part A — Data Preparation**
- Parsed `Timestamp IST` (DD-MM-YYYY HH:MM) → Python `datetime`, extracted date.
- Trimmed Fear/Greed index to the overlapping date window.
- Built daily metrics per trader: PnL, win rate, trade count, avg size USD, long/short ratio.
- Merged on `date` → 2,340 trader-day records across 32 unique accounts.

**Part B — Analysis**
- Compared avg PnL, win rate, trade count, size, and directional bias across all 5 sentiment classes.
- Segmented traders into: High-Size vs Low-Size, Frequent vs Infrequent, Consistent Winner vs Inconsistent.
- Generated 6 charts covering performance, behavior, distributions, heatmaps, time-series, and rankings.

**Bonus**
- Random Forest classifier (150 trees, max depth 6) predicting next-day PnL bucket (Loss/Flat/Profit) — **63% accuracy**.
- KMeans clustering (k=3) identifying 3 behavioral archetypes.

---

## Key Insights

**Insight 1 — Fear Outperforms Greed (Counter-Intuitive)**  
Average daily PnL is highest on *Fear* days ($5,329) and *Extreme Greed* days ($5,162), and lowest on *Greed* days ($3,318). This cohort of Hyperliquid traders behaves as **contrarians** — they exploit volatility rather than follow sentiment momentum.

**Insight 2 — Traders Are Most Active in Fear**  
Trade frequency peaks at ~98 trades/day during Fear vs ~76 trades/day during Extreme Greed. Average position sizes are also larger in Fear ($8,976) vs Greed ($6,428). These traders treat fear as a buying/trading opportunity.

**Insight 3 — Directional Bias Shifts with Sentiment**  
Long ratio is 52% on Fear days vs 47% on Greed days. Traders lean long during fear (mean-reversion) and short during greed (fade the rally). This directional edge is consistent across all 32 accounts.

**Insight 4 — Consistent Winners Thrive in Neutral Markets**  
Traders with Sharpe > 0.5 and win rate > 50% (6 out of 32) generate $6,940 avg daily PnL during Neutral sentiment — far above their Fear ($4,139) and Greed ($3,375) performance. Volatile sentiment regimes add noise that hurts their precision-based strategies.

**Insight 5 — Sentiment Is a Useful Predictive Feature**  
`sentiment_score` ranks among the top 3 features in the Random Forest model. Combined with behavioral metrics (trade count, win rate, daily PnL), it achieves 63% accuracy in predicting whether tomorrow's PnL will be a Loss, Flat, or Profit.

---

## Strategy Recommendations

**Rule 1 — Contrarian Sizing: More Trades, Smaller Positions in Fear**  
During Extreme Fear (score < 25), increase trade frequency by up to 30% but keep individual position sizes smaller to manage elevated tail risk. During Greed (score > 60), reduce overall trade count. Only Consistent-Winner accounts (Sharpe > 0.5, win rate > 50%) should maintain full position sizing in Greed.

**Rule 2 — Directional Bias Shift with Sentiment**  
On Fear days, tilt directional exposure LONG (mean-reversion bet on oversold conditions).  
On Greed days, tilt SHORT or stay neutral — fade the rally. Infrequent traders should not chase momentum in Greed: data shows their PnL deteriorates when they increase activity in high-sentiment environments.

---

## Output Charts

| Chart | Description |
|-------|-------------|
| `chart1_performance_by_sentiment.png` | Avg PnL and win rate across 5 sentiment classes |
| `chart2_behavior_by_sentiment.png` | Trade count, size, long ratio by sentiment |
| `chart3_pnl_distribution.png` | Box plot + density: Fear vs Greed PnL |
| `chart4_segment_heatmap.png` | Segment × sentiment avg PnL heatmap |
| `chart5_timeseries.png` | Sentiment score vs aggregate PnL over time |
| `chart6_top_traders.png` | Top 10 traders by Sharpe and total PnL |
| `chart7_feature_importance.png` | Random Forest feature importances |
| `chart8_trader_clusters.png` | KMeans behavioral archetypes scatter |
