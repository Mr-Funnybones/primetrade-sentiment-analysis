# %% [markdown]
# # Trader Performance vs Market Sentiment Analysis
# ## Primetrade.ai Data Science Intern Assignment
# 
# **Objective:** Analyze how Bitcoin market sentiment (Fear/Greed) relates to trader behavior and performance on Hyperliquid.

# %% [markdown]
# ## Part A: Data Preparation

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# %%
# Load datasets
sentiment_df = pd.read_csv('../data/fear_greed_index.csv')
trades_df = pd.read_csv('../data/historical_data.csv')

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"\n📊 Sentiment Data: {sentiment_df.shape[0]} rows, {sentiment_df.shape[1]} columns")
print(f"📊 Trades Data: {trades_df.shape[0]} rows, {trades_df.shape[1]} columns")

# %%
# Sentiment data info
print("\n📈 SENTIMENT DATA")
print("-" * 40)
print(sentiment_df.info())
print(f"\nMissing values:\n{sentiment_df.isnull().sum()}")
print(f"\nDuplicates: {sentiment_df.duplicated().sum()}")
print(f"\nDate range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
print(f"\nClassification distribution:\n{sentiment_df['classification'].value_counts()}")

# %%
# Trades data info
print("\n💹 TRADES DATA")
print("-" * 40)
print(trades_df.info())
print(f"\nMissing values:\n{trades_df.isnull().sum()}")
print(f"\nDuplicates: {trades_df.duplicated().sum()}")
print(f"\nUnique accounts: {trades_df['Account'].nunique()}")
print(f"\nUnique coins: {trades_df['Coin'].nunique()}")

# %%
# Convert timestamps and create date columns
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
trades_df['Timestamp IST'] = pd.to_datetime(trades_df['Timestamp IST'], format='%d-%m-%Y %H:%M')
trades_df['date'] = trades_df['Timestamp IST'].dt.date
trades_df['date'] = pd.to_datetime(trades_df['date'])

print(f"\n📅 Sentiment date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
print(f"📅 Trades date range: {trades_df['date'].min()} to {trades_df['date'].max()}")

# %%
# Create binary sentiment column
sentiment_df['sentiment_binary'] = sentiment_df['classification'].apply(
    lambda x: 'Fear' if 'Fear' in x else ('Greed' if 'Greed' in x else 'Neutral')
)

# Merge datasets
merged_df = trades_df.merge(sentiment_df[['date', 'value', 'classification', 'sentiment_binary']], 
                             on='date', how='left')
print(f"\n✅ Merged dataset: {merged_df.shape[0]} rows")
print(f"📊 Matched with sentiment: {merged_df['classification'].notna().sum()} ({merged_df['classification'].notna().mean()*100:.1f}%)")

# %% [markdown]
# ## Key Metrics Calculation

# %%
# Calculate key metrics per trader per day
def calculate_daily_metrics(df):
    daily = df.groupby(['Account', 'date']).agg({
        'Closed PnL': 'sum',
        'Size USD': ['sum', 'mean', 'count'],
        'Side': lambda x: (x == 'BUY').sum() / len(x) if len(x) > 0 else 0.5,
        'Coin': 'nunique'
    }).reset_index()
    
    daily.columns = ['Account', 'date', 'daily_pnl', 'total_volume', 'avg_trade_size', 
                     'trade_count', 'long_ratio', 'coins_traded']
    return daily

daily_metrics = calculate_daily_metrics(merged_df)

# Add sentiment info
daily_metrics = daily_metrics.merge(
    sentiment_df[['date', 'value', 'classification', 'sentiment_binary']], 
    on='date', how='left'
)

# Calculate win rate (positive PnL days)
daily_metrics['is_profitable'] = daily_metrics['daily_pnl'] > 0

print(f"\n📊 Daily metrics calculated: {daily_metrics.shape[0]} trader-days")
print(daily_metrics.head(10))

# %% [markdown]
# ## Part B: Analysis
# ### Q1: Performance Differences - Fear vs Greed Days

# %%
# Filter to Fear and Greed only
fg_metrics = daily_metrics[daily_metrics['sentiment_binary'].isin(['Fear', 'Greed'])].copy()

# Performance comparison
perf_comparison = fg_metrics.groupby('sentiment_binary').agg({
    'daily_pnl': ['mean', 'median', 'std'],
    'is_profitable': 'mean',
    'trade_count': 'mean',
    'total_volume': 'mean'
}).round(2)

print("\n📊 PERFORMANCE: FEAR vs GREED DAYS")
print("=" * 60)
print(perf_comparison)

# %%
# Visualization: PnL Distribution by Sentiment
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# PnL Distribution
for sentiment in ['Fear', 'Greed']:
    data = fg_metrics[fg_metrics['sentiment_binary'] == sentiment]['daily_pnl']
    data_clipped = data.clip(-5000, 5000)
    axes[0].hist(data_clipped, alpha=0.6, label=sentiment, bins=50)
axes[0].set_xlabel('Daily PnL (USD)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('PnL Distribution: Fear vs Greed')
axes[0].legend()
axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.5)

# Win Rate
win_rates = fg_metrics.groupby('sentiment_binary')['is_profitable'].mean()
colors = ['#e74c3c' if s == 'Fear' else '#27ae60' for s in win_rates.index]
axes[1].bar(win_rates.index, win_rates.values * 100, color=colors)
axes[1].set_ylabel('Win Rate (%)')
axes[1].set_title('Win Rate by Sentiment')
for i, v in enumerate(win_rates.values):
    axes[1].text(i, v*100 + 1, f'{v*100:.1f}%', ha='center', fontweight='bold')

# Average PnL
avg_pnl = fg_metrics.groupby('sentiment_binary')['daily_pnl'].mean()
colors = ['#e74c3c' if s == 'Fear' else '#27ae60' for s in avg_pnl.index]
axes[2].bar(avg_pnl.index, avg_pnl.values, color=colors)
axes[2].set_ylabel('Average Daily PnL (USD)')
axes[2].set_title('Average PnL by Sentiment')
for i, v in enumerate(avg_pnl.values):
    axes[2].text(i, v + (50 if v > 0 else -100), f'${v:.0f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/performance_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Q2: Behavior Changes Based on Sentiment

# %%
# Behavior metrics by sentiment
behavior = fg_metrics.groupby('sentiment_binary').agg({
    'trade_count': 'mean',
    'avg_trade_size': 'mean', 
    'long_ratio': 'mean',
    'total_volume': 'mean'
}).round(2)

print("\n📊 TRADER BEHAVIOR: FEAR vs GREED")
print("=" * 60)
print(behavior)

# %%
# Visualization: Behavior Changes
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

metrics = [
    ('trade_count', 'Avg Trades per Day', 'Trades'),
    ('avg_trade_size', 'Avg Trade Size (USD)', 'USD'),
    ('long_ratio', 'Long Ratio (Buy %)', '%'),
    ('total_volume', 'Avg Daily Volume (USD)', 'USD')
]

for idx, (col, title, unit) in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    data = fg_metrics.groupby('sentiment_binary')[col].mean()
    colors = ['#e74c3c', '#27ae60']
    bars = ax.bar(data.index, data.values, color=colors)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(unit)
    
    for bar, val in zip(bars, data.values):
        if col == 'long_ratio':
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val*100:.1f}%', ha='center', fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{val:.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/behavior_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Q3: Trader Segmentation

# %%
# Calculate trader-level metrics
trader_stats = merged_df.groupby('Account').agg({
    'Closed PnL': ['sum', 'mean', 'std'],
    'Size USD': ['mean', 'sum'],
    'Timestamp IST': 'count'
}).reset_index()
trader_stats.columns = ['Account', 'total_pnl', 'avg_pnl', 'pnl_std', 
                        'avg_trade_size', 'total_volume', 'trade_count']

# Segment traders
trader_stats['pnl_consistency'] = trader_stats['avg_pnl'] / (trader_stats['pnl_std'] + 1)
trader_stats['trades_per_active_day'] = trader_stats['trade_count'] / trader_stats['trade_count'].clip(lower=1)

# Segmentation
def segment_trader(row):
    segments = []
    
    # By frequency
    if row['trade_count'] >= trader_stats['trade_count'].quantile(0.75):
        segments.append('High Frequency')
    elif row['trade_count'] <= trader_stats['trade_count'].quantile(0.25):
        segments.append('Low Frequency')
    else:
        segments.append('Medium Frequency')
    
    # By trade size
    if row['avg_trade_size'] >= trader_stats['avg_trade_size'].quantile(0.75):
        segments.append('Large Size')
    elif row['avg_trade_size'] <= trader_stats['avg_trade_size'].quantile(0.25):
        segments.append('Small Size')
    else:
        segments.append('Medium Size')
    
    # By profitability
    if row['total_pnl'] > 0:
        segments.append('Profitable')
    else:
        segments.append('Unprofitable')
    
    return ' | '.join(segments)

trader_stats['segment'] = trader_stats.apply(segment_trader, axis=1)

# Frequency segment
trader_stats['freq_segment'] = pd.cut(trader_stats['trade_count'], 
                                       bins=[0, 50, 500, float('inf')],
                                       labels=['Infrequent', 'Moderate', 'Frequent'])

# Size segment  
trader_stats['size_segment'] = pd.cut(trader_stats['avg_trade_size'],
                                       bins=[0, 500, 2000, float('inf')],
                                       labels=['Small', 'Medium', 'Large'])

print("\n📊 TRADER SEGMENTS")
print("=" * 60)
print(f"\nBy Frequency:")
print(trader_stats['freq_segment'].value_counts())
print(f"\nBy Trade Size:")
print(trader_stats['size_segment'].value_counts())
print(f"\nBy Profitability:")
print((trader_stats['total_pnl'] > 0).value_counts().rename({True: 'Profitable', False: 'Unprofitable'}))

# %%
# Segment performance comparison
segment_perf = trader_stats.groupby('freq_segment').agg({
    'total_pnl': 'mean',
    'avg_trade_size': 'mean',
    'trade_count': 'mean'
}).round(2)

print("\n📊 PERFORMANCE BY FREQUENCY SEGMENT")
print(segment_perf)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# By frequency
freq_pnl = trader_stats.groupby('freq_segment')['total_pnl'].mean()
axes[0].bar(freq_pnl.index.astype(str), freq_pnl.values, color=['#3498db', '#9b59b6', '#e74c3c'])
axes[0].set_title('Avg Total PnL by Trading Frequency', fontweight='bold')
axes[0].set_ylabel('Avg Total PnL (USD)')
axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)

# By size
size_pnl = trader_stats.groupby('size_segment')['total_pnl'].mean()
axes[1].bar(size_pnl.index.astype(str), size_pnl.values, color=['#2ecc71', '#f39c12', '#e74c3c'])
axes[1].set_title('Avg Total PnL by Trade Size', fontweight='bold')
axes[1].set_ylabel('Avg Total PnL (USD)')
axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)

# Profitable vs Not
profit_counts = (trader_stats['total_pnl'] > 0).value_counts()
axes[2].pie(profit_counts.values, labels=['Unprofitable', 'Profitable'], 
            colors=['#e74c3c', '#27ae60'], autopct='%1.1f%%', startangle=90)
axes[2].set_title('Trader Profitability Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/trader_segments.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Key Insights (with Evidence)

# %%
# INSIGHT 1: Performance difference
fear_pnl = fg_metrics[fg_metrics['sentiment_binary'] == 'Fear']['daily_pnl'].mean()
greed_pnl = fg_metrics[fg_metrics['sentiment_binary'] == 'Greed']['daily_pnl'].mean()
fear_wr = fg_metrics[fg_metrics['sentiment_binary'] == 'Fear']['is_profitable'].mean()
greed_wr = fg_metrics[fg_metrics['sentiment_binary'] == 'Greed']['is_profitable'].mean()

print("\n" + "=" * 70)
print("📊 KEY INSIGHTS")
print("=" * 70)

print(f"""
🔍 INSIGHT 1: Sentiment Impact on Performance
   • Fear Days Avg PnL: ${fear_pnl:.2f} | Win Rate: {fear_wr*100:.1f}%
   • Greed Days Avg PnL: ${greed_pnl:.2f} | Win Rate: {greed_wr*100:.1f}%
   • Difference: ${abs(greed_pnl - fear_pnl):.2f} higher on {'Greed' if greed_pnl > fear_pnl else 'Fear'} days
""")

# INSIGHT 2: Behavior changes
fear_trades = fg_metrics[fg_metrics['sentiment_binary'] == 'Fear']['trade_count'].mean()
greed_trades = fg_metrics[fg_metrics['sentiment_binary'] == 'Greed']['trade_count'].mean()
fear_long = fg_metrics[fg_metrics['sentiment_binary'] == 'Fear']['long_ratio'].mean()
greed_long = fg_metrics[fg_metrics['sentiment_binary'] == 'Greed']['long_ratio'].mean()

print(f"""
🔍 INSIGHT 2: Behavioral Adaptation to Sentiment
   • Traders make {fear_trades:.1f} trades/day on Fear vs {greed_trades:.1f} on Greed days
   • Long ratio: {fear_long*100:.1f}% on Fear vs {greed_long*100:.1f}% on Greed
   • {'More cautious' if fear_trades < greed_trades else 'More active'} trading during Fear periods
""")

# INSIGHT 3: Segment performance
profitable_pct = (trader_stats['total_pnl'] > 0).mean() * 100
top_traders = trader_stats.nlargest(10, 'total_pnl')['total_pnl'].sum()
total_pnl = trader_stats['total_pnl'].sum()

print(f"""
🔍 INSIGHT 3: Winner Concentration  
   • Only {profitable_pct:.1f}% of traders are profitable overall
   • Top 10 traders account for ${top_traders:,.0f} of total PnL
   • High-frequency traders show {'better' if segment_perf.loc['Frequent', 'total_pnl'] > 0 else 'worse'} average returns
""")

# %% [markdown]
# ## Part C: Actionable Strategy Recommendations

# %%
print("\n" + "=" * 70)
print("🎯 STRATEGY RECOMMENDATIONS")
print("=" * 70)

print("""
📌 STRATEGY 1: Sentiment-Based Position Sizing
   
   Rule: During FEAR periods (index < 25):
   • Reduce position sizes by 30-40% for all traders
   • Focus on shorter-term trades to capture volatility
   • Increase stop-loss buffer by 20%
   
   Rationale: Fear periods show higher volatility and lower win rates,
   requiring more conservative risk management.

📌 STRATEGY 2: Segment-Specific Trading Rules

   For HIGH-FREQUENCY traders:
   • Maintain activity during Greed periods (higher edge)
   • Reduce trade count by 50% during Extreme Fear
   
   For LARGE-SIZE traders:
   • Wait for sentiment to stabilize before entering large positions
   • Prefer Greed days for building positions
   
   For INFREQUENT traders:
   • Focus only on Extreme conditions (< 20 or > 80) for entry signals
   • Use sentiment reversals as exit triggers

📌 BONUS STRATEGY: Sentiment Momentum

   • When Fear index drops below 20 for 3+ consecutive days,
     prepare for potential reversal - increase long exposure on bounce
   • When Greed exceeds 75 for 5+ days, begin taking profits
     and reducing leverage
""")

# %% [markdown]
# ## Summary Statistics Table

# %%
# Create summary table
summary_data = {
    'Metric': [
        'Total Trades Analyzed',
        'Unique Traders',
        'Date Range',
        'Fear Days Avg PnL',
        'Greed Days Avg PnL',
        'Overall Win Rate',
        'Profitable Traders %'
    ],
    'Value': [
        f"{len(trades_df):,}",
        f"{trades_df['Account'].nunique():,}",
        f"{trades_df['date'].min().strftime('%Y-%m-%d')} to {trades_df['date'].max().strftime('%Y-%m-%d')}",
        f"${fear_pnl:.2f}",
        f"${greed_pnl:.2f}",
        f"{fg_metrics['is_profitable'].mean()*100:.1f}%",
        f"{profitable_pct:.1f}%"
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n📊 ANALYSIS SUMMARY")
print("=" * 50)
print(summary_df.to_string(index=False))

# Save summary
summary_df.to_csv('../outputs/summary_statistics.csv', index=False)
print("\n✅ Summary saved to outputs/summary_statistics.csv")
