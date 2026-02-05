# Trader Performance vs Market Sentiment Analysis

Analysis of how Bitcoin market sentiment relates to trader behavior on Hyperliquid.

## Overview

This project explores the relationship between the Fear & Greed Index and trading patterns on Hyperliquid DEX. The goal is to find actionable insights that could help traders make better decisions based on market sentiment.

## Files

```
├── data/
│   ├── fear_greed_index.csv    # Bitcoin sentiment data
│   └── historical_data.csv      # Trading records
├── notebooks/
│   └── analysis.py              # Main analysis
├── outputs/                     # Charts and results
├── SUMMARY.md                   # 1-page write-up
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
cd notebooks
python analysis.py
```

The script generates charts in `outputs/` and prints insights to console.

## Quick Findings

- **Fear vs Greed**: Traders show higher win rates on Greed days (64%) vs Fear days (60%)
- **Behavior shifts**: Trade frequency increases during Fear periods, suggesting reactive trading
- **Concentration**: A small group of traders captures most profits

## Author

Aditya
