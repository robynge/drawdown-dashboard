"""Peer group analysis for stocks"""
import pandas as pd
import numpy as np
from config import START_DATE, END_DATE
from data_loader import load_industry_info, load_r3000_holdings

def get_stock_peer_group(stock_ticker, source='ark'):
    """Get peer group for a stock based on GICS industry"""
    industry_map = load_industry_info(source)
    stock_industry = industry_map.get(stock_ticker)

    if not stock_industry:
        return None, []

    peers = [ticker for ticker, industry in industry_map.items()
             if industry == stock_industry and ticker != stock_ticker]

    return stock_industry, peers

def calculate_peer_metrics_mv(stock_ticker, holdings_df, r3000_df, etf=None):
    """Calculate market-value weighted peer group metrics"""
    industry, peers = get_stock_peer_group(stock_ticker, source='ark')

    if not industry or not peers:
        return None

    # Filter period
    stock_data = holdings_df[holdings_df['Ticker'] == stock_ticker].copy()
    stock_data = stock_data[(stock_data['Date'] >= START_DATE) & (stock_data['Date'] <= END_DATE)]

    if etf:
        stock_data = stock_data[stock_data['ETF'] == etf]

    if len(stock_data) == 0:
        return None

    # Get peer data from R3000
    peer_data = r3000_df[r3000_df['Ticker'].isin(peers)].copy()
    peer_data = peer_data[(peer_data['Date'] >= START_DATE) & (peer_data['Date'] <= END_DATE)]

    if len(peer_data) == 0:
        return None

    # Calculate weighted average by date
    peer_grouped = peer_data.groupby('Date').apply(
        lambda x: pd.Series({
            'peer_avg_mv': (x['Market Value'] * x['Weight (%)']).sum() / x['Weight (%)'].sum()
                           if x['Weight (%)'].sum() > 0 else np.nan
        })
    ).reset_index()

    # Merge with stock data
    merged = pd.merge(stock_data[['Date', 'Market Value']], peer_grouped, on='Date', how='inner')

    return {
        'industry': industry,
        'num_peers': len(peers),
        'data': merged
    }

def calculate_peer_metrics_weighted(stock_ticker, holdings_df, r3000_df, etf=None):
    """Calculate equal-weighted peer group metrics"""
    industry, peers = get_stock_peer_group(stock_ticker, source='ark')

    if not industry or not peers:
        return None

    stock_data = holdings_df[holdings_df['Ticker'] == stock_ticker].copy()
    stock_data = stock_data[(stock_data['Date'] >= START_DATE) & (stock_data['Date'] <= END_DATE)]

    if etf:
        stock_data = stock_data[stock_data['ETF'] == etf]

    if len(stock_data) == 0:
        return None

    peer_data = r3000_df[r3000_df['Ticker'].isin(peers)].copy()
    peer_data = peer_data[(peer_data['Date'] >= START_DATE) & (peer_data['Date'] <= END_DATE)]

    if len(peer_data) == 0:
        return None

    # Equal-weighted average
    peer_grouped = peer_data.groupby('Date')['Market Value'].mean().reset_index()
    peer_grouped.rename(columns={'Market Value': 'peer_avg_mv'}, inplace=True)

    merged = pd.merge(stock_data[['Date', 'Market Value']], peer_grouped, on='Date', how='inner')

    return {
        'industry': industry,
        'num_peers': len(peers),
        'data': merged
    }

def analyze_all_peer_groups(source='ark'):
    """Get summary of all GICS industry groups"""
    industry_map = load_industry_info(source)
    r3000_df = load_r3000_holdings()

    industry_stats = {}
    for ticker, industry in industry_map.items():
        if industry not in industry_stats:
            industry_stats[industry] = {'tickers': [], 'mv_data': []}
        industry_stats[industry]['tickers'].append(ticker)

    # Calculate aggregate metrics per industry
    for industry in industry_stats:
        tickers = industry_stats[industry]['tickers']
        industry_data = r3000_df[r3000_df['Ticker'].isin(tickers)].copy()
        industry_data = industry_data[(industry_data['Date'] >= START_DATE) &
                                       (industry_data['Date'] <= END_DATE)]

        if len(industry_data) > 0:
            industry_stats[industry]['mv_data'] = industry_data

    return industry_stats
