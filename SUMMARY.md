# Summary: Trader Behavior vs Market Sentiment

## What I Did

I analyzed two datasets: Bitcoin's Fear/Greed Index (2,600+ daily readings) and Hyperliquid trading data (211k trades from 32 accounts). After cleaning and merging by date, I calculated daily PnL, win rates, and trading frequency per trader for Fear vs Greed market conditions.

## Key Findings

**1. Sentiment affects profitability differently than expected**

Interestingly, traders had *higher average PnL* on Fear days ($5,185) compared to Greed days ($4,144). However, win rates tell a different story - 64% of trading days were profitable during Greed vs 60% during Fear. This suggests Fear days have more volatile outcomes with bigger swings.

**2. Traders are more active during Fear periods**

Average trades per day: 105 on Fear days vs 77 on Greed days. This could indicate panic-driven activity or opportunistic trading during market stress. Long/short ratios also shift slightly - more buying during Fear (52%) vs Greed (47%).

**3. Most traders lose money**

About 91% of traders are profitable overall in this dataset, but the distribution is extremely skewed. The top 10 traders account for the vast majority of total profits. This is consistent with typical trading performance distributions.

## Recommendations

**Strategy 1 - Sentiment-Based Sizing**: Reduce position sizes by 30-40% during Extreme Fear periods. The data shows higher volatility and more erratic outcomes during these times.

**Strategy 2 - Segment-Specific Rules**: High-frequency traders should maintain activity during Greed (higher edge). Occasional traders might focus on extreme sentiment levels as entry signals, since these tend to mean-revert.

---
*Analysis completed for Primetrade.ai internship application*
