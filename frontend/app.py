import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import asyncio

# Import visualization components
from M11.Visulization.VIS import FinancialVisualizer

# Configure the app
st.set_page_config(page_title="Financial Analytics Platform", layout="wide")

# API endpoint
API_BASE_URL = "http://localhost:8000"

def main():
    st.title("Financial Analytics Platform")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Navigate", 
        ["Market Data", "Social Analysis", "News", "Trading Strategies", "Visualization"]
    )
    
    if page == "Market Data":
        display_market_data()
    elif page == "Social Analysis":
        display_social_analysis()
    elif page == "News":
        display_news()
    elif page == "Trading Strategies":
        display_trading_strategies()
    elif page == "Visualization":
        display_visualization()

def display_market_data():
    st.header("Market Data Analysis")
    
    symbol = st.text_input("Enter Stock Symbol", value="AAPL")
    if st.button("Fetch Data"):
        response = requests.get(f"{API_BASE_URL}/market-data/{symbol}")
        if response.status_code == 200:
            data = pd.DataFrame(response.json())
            
            # Create visualizer
            viz = FinancialVisualizer(data)
            
            # Display candlestick chart
            fig = viz.plot_candlestick(engine='plotly')
            st.plotly_chart(fig)
            
            # Display technical analysis
            tech_fig = viz.plot_technical_analysis()
            st.plotly_chart(tech_fig)

def display_social_analysis():
    st.header("Social Media Analysis")
    
    symbol = st.text_input("Enter Stock Symbol", value="AAPL")
    if st.button("Analyze Social Sentiment"):
        response = requests.get(f"{API_BASE_URL}/social-metrics/{symbol}")
        if response.status_code == 200:
            metrics = response.json()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Momentum Score", f"{metrics['momentum_score']:.2f}")
            with col2:
                st.metric("Reddit Sentiment", f"{metrics['reddit_sentiment']['average_sentiment']:.2f}")
            with col3:
                st.metric("StockTwits Bullish Ratio", f"{metrics['stocktwits_metrics']['bullish_ratio']:.2%}")

def display_news():
    st.header("Financial News")
    
    if st.button("Refresh News"):
        response = requests.get(f"{API_BASE_URL}/news")
        if response.status_code == 200:
            news_items = response.json()
            
            for item in news_items:
                with st.expander(item['title']):
                    st.write(item['summary'])

def display_trading_strategies():
    st.header("Trading Strategies")
    
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL")
    with col2:
        strategy = st.selectbox("Select Strategy", ["momentum", "mean_reversion"])
    
    if st.button("Analyze Strategy"):
        response = requests.post(
            f"{API_BASE_URL}/analyze-strategy",
            params={"symbol": symbol, "strategy_type": strategy}
        )
        if response.status_code == 200:
            signal = response.json()
            st.success(f"Trading Signal: {signal}")

def display_visualization():
    st.header("Advanced Visualization")
    
    symbol = st.text_input("Enter Stock Symbol", value="AAPL")
    viz_type = st.selectbox(
        "Select Visualization", 
        ["Interactive Analysis", "Statistical Analysis", "Technical Dashboard"]
    )
    
    if st.button("Generate Visualization"):
        response = requests.get(f"{API_BASE_URL}/market-data/{symbol}")
        if response.status_code == 200:
            data = pd.DataFrame(response.json())
            viz = FinancialVisualizer(data)
            
            if viz_type == "Interactive Analysis":
                fig = viz.plot_interactive_analysis(engine='plotly')
            elif viz_type == "Statistical Analysis":
                fig = viz.plot_statistical_analysis()
            else:
                fig = viz.plot_technical_analysis()
                
            st.plotly_chart(fig)

if __name__ == "__main__":
    main() 