"""Plotly chart builders"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def create_price_chart_with_drawdowns(price_df, drawdown_df, title="Price with Drawdowns"):
    """Create price chart with drawdown shaded regions"""
    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=price_df['Date'],
        y=price_df['Close'],
        mode='lines',
        name='Price',
        line=dict(color='black', width=2)
    ))

    # Drawdown regions
    colors = ['rgba(255, 99, 71, 0.3)', 'rgba(255, 165, 0, 0.3)', 'rgba(255, 215, 0, 0.3)',
              'rgba(144, 238, 144, 0.3)', 'rgba(173, 216, 230, 0.3)']

    historical_dd = drawdown_df[drawdown_df['rank'] != 'Current'].head(5)
    for idx, (_, row) in enumerate(historical_dd.iterrows()):
        fig.add_vrect(
            x0=row['peak_date'],
            x1=row['trough_date'],
            fillcolor=colors[idx % len(colors)],
            layer="below",
            line_width=0,
            annotation_text=f"DD{row['rank']}: {row['depth_pct']:.1f}%",
            annotation_position="top left"
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=500
    )

    return fig

def create_mv_comparison_chart(stock_mv, peer_avg_mv, dates, title="Market Value Comparison"):
    """Create MV comparison chart for stock vs peers"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=stock_mv,
        mode='lines',
        name='Stock MV',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=peer_avg_mv,
        mode='lines',
        name='Peer Avg MV',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Market Value ($)",
        hovermode='x unified',
        height=500
    )

    return fig

def create_drawdown_heatmap(drawdown_data_dict, metric='depth_pct'):
    """Create heatmap of drawdowns across multiple assets"""
    data = []
    for ticker, dd_df in drawdown_data_dict.items():
        if len(dd_df) > 0:
            max_dd = dd_df[dd_df['rank'] != 'Current'][metric].min() if len(dd_df[dd_df['rank'] != 'Current']) > 0 else 0
            data.append({'ticker': ticker, 'max_drawdown': max_dd})

    df = pd.DataFrame(data)
    df = df.sort_values('max_drawdown')

    fig = px.bar(
        df,
        x='ticker',
        y='max_drawdown',
        title='Maximum Drawdown by Asset',
        color='max_drawdown',
        color_continuous_scale='Reds_r'
    )

    fig.update_layout(
        xaxis_title="Ticker",
        yaxis_title="Max Drawdown (%)",
        height=500
    )

    return fig

def create_multi_stock_comparison(price_data_dict, title="Multi-Stock Price Comparison"):
    """Compare multiple stocks on normalized price chart"""
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    for idx, (ticker, price_df) in enumerate(price_data_dict.items()):
        # Normalize to 100 at start
        normalized = (price_df['Close'] / price_df['Close'].iloc[0]) * 100

        fig.add_trace(go.Scatter(
            x=price_df['Date'],
            y=normalized,
            mode='lines',
            name=ticker,
            line=dict(color=colors[idx % len(colors)], width=2)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Normalized Price (Base 100)",
        hovermode='x unified',
        height=500
    )

    return fig

def create_peer_group_chart(industry_data, industry_name):
    """Create chart showing all stocks in a peer group"""
    fig = go.Figure()

    colors = px.colors.qualitative.Safe

    for idx, ticker in enumerate(industry_data['Ticker'].unique()):
        ticker_data = industry_data[industry_data['Ticker'] == ticker]

        fig.add_trace(go.Scatter(
            x=ticker_data['Date'],
            y=ticker_data['Market Value'],
            mode='lines',
            name=ticker,
            line=dict(color=colors[idx % len(colors)], width=1.5)
        ))

    fig.update_layout(
        title=f"{industry_name} - All Stocks",
        xaxis_title="Date",
        yaxis_title="Market Value ($)",
        hovermode='x unified',
        height=600,
        showlegend=True
    )

    return fig
