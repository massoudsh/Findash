"""
Streamlit Dashboard - BTC Price Pub/Sub Demo
Demonstrates Redis Pub/Sub mechanism for real-time BTC price updates
Runs on port 8500
"""

import streamlit as st
import redis
import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="BTC Price Pub/Sub Demo",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #f7931a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .price-display {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
    }
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem;
    }
    .status-connected {
        background-color: #10b981;
        color: white;
    }
    .status-disconnected {
        background-color: #ef4444;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Redis connection
@st.cache_resource
def get_redis_client():
    """Get Redis client connection"""
    try:
        redis_client = redis.from_url(
            'redis://localhost:6379/0',
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        redis_client.ping()
        return redis_client, None
    except Exception as e:
        return None, str(e)

# Initialize session state
if 'btc_prices' not in st.session_state:
    st.session_state.btc_prices = []
if 'messages_received' not in st.session_state:
    st.session_state.messages_received = 0
if 'subscriber_running' not in st.session_state:
    st.session_state.subscriber_running = False
if 'pubsub_client' not in st.session_state:
    st.session_state.pubsub_client = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

def subscribe_to_btc_updates(redis_client):
    """Subscribe to BTC price updates via Redis Pub/Sub"""
    pubsub = redis_client.pubsub()
    pubsub.subscribe('btc_price_updates')
    st.session_state.pubsub_client = pubsub
    st.session_state.subscriber_running = True
    
    st.sidebar.success("✅ Subscribed to 'btc_price_updates' channel")
    
    # Listen for messages
    for message in pubsub.listen():
        if not st.session_state.subscriber_running:
            break
            
        if message['type'] == 'message':
            try:
                data = json.loads(message['data'])
                
                # Update session state
                st.session_state.btc_prices.append({
                    'price': data.get('price', 0),
                    'change_24h': data.get('change_24h', 0),
                    'timestamp': datetime.now(),
                    'source': data.get('source', 'unknown')
                })
                
                # Keep only last 100 prices
                if len(st.session_state.btc_prices) > 100:
                    st.session_state.btc_prices.pop(0)
                
                st.session_state.messages_received += 1
                st.session_state.last_update = datetime.now()
                
            except Exception as e:
                st.sidebar.error(f"Error processing message: {e}")

def unsubscribe_from_btc_updates():
    """Unsubscribe from BTC price updates"""
    if st.session_state.pubsub_client:
        st.session_state.pubsub_client.unsubscribe('btc_price_updates')
        st.session_state.pubsub_client.close()
        st.session_state.pubsub_client = None
    st.session_state.subscriber_running = False
    st.sidebar.warning("❌ Unsubscribed from 'btc_price_updates' channel")

# Main UI
st.markdown('<div class="main-header">₿ BTC Price Pub/Sub Demo</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("🔧 Controls")
st.sidebar.markdown("---")

# Get Redis connection
redis_client, redis_error = get_redis_client()

if redis_error:
    st.sidebar.error(f"❌ Redis Connection Error: {redis_error}")
    st.sidebar.info("💡 Make sure Redis is running on localhost:6379")
else:
    st.sidebar.success("✅ Connected to Redis")

# Subscription controls
st.sidebar.markdown("### 📡 Pub/Sub Subscription")

if redis_client:
    if not st.session_state.subscriber_running:
        if st.sidebar.button("▶️ Start Subscribing", type="primary", use_container_width=True):
            # Start subscription in a thread
            thread = threading.Thread(
                target=subscribe_to_btc_updates,
                args=(redis_client,),
                daemon=True
            )
            thread.start()
            time.sleep(0.5)  # Give thread time to start
            st.rerun()
    else:
        if st.sidebar.button("⏹️ Stop Subscribing", type="secondary", use_container_width=True):
            unsubscribe_from_btc_updates()
            st.rerun()

# Status display
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Status")

if st.session_state.subscriber_running:
    st.sidebar.markdown('<span class="status-badge status-connected">🟢 SUBSCRIBED</span>', unsafe_allow_html=True)
    st.sidebar.metric("Messages Received", st.session_state.messages_received)
    if st.session_state.last_update:
        time_since = (datetime.now() - st.session_state.last_update).total_seconds()
        st.sidebar.metric("Last Update", f"{time_since:.1f}s ago")
else:
    st.sidebar.markdown('<span class="status-badge status-disconnected">🔴 NOT SUBSCRIBED</span>', unsafe_allow_html=True)

# Channel info
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Channel Info")
st.sidebar.code("Channel: btc_price_updates\nPublisher: Celery Task\nFrequency: Every 5 seconds")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📈 Real-time BTC Price Updates")
    
    if not st.session_state.subscriber_running:
        st.info("👆 Click 'Start Subscribing' in the sidebar to begin receiving real-time updates via Redis Pub/Sub")
    
    if st.session_state.btc_prices:
        latest_price = st.session_state.btc_prices[-1]
        
        # Price display
        price_color = "#10b981" if latest_price['change_24h'] >= 0 else "#ef4444"
        change_symbol = "📈" if latest_price['change_24h'] >= 0 else "📉"
        
        st.markdown(f"""
        <div class="price-display" style="background: linear-gradient(135deg, {price_color} 0%, #667eea 100%);">
            ${latest_price['price']:,.2f}
            <br>
            <span style="font-size: 1.5rem;">
                {change_symbol} {latest_price['change_24h']:+.2f}%
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Price chart
        if len(st.session_state.btc_prices) > 1:
            prices = [p['price'] for p in st.session_state.btc_prices]
            timestamps = [p['timestamp'] for p in st.session_state.btc_prices]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=prices,
                mode='lines+markers',
                name='BTC Price',
                line=dict(color='#f7931a', width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="BTC Price Over Time (Pub/Sub Updates)",
                xaxis_title="Time",
                yaxis_title="Price (USD)",
                height=400,
                hovermode='x unified',
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⏳ Waiting for price updates...")

with col2:
    st.markdown("### 📊 Metrics")
    
    # Metrics cards
    if st.session_state.btc_prices:
        latest = st.session_state.btc_prices[-1]
        
        st.metric("Current Price", f"${latest['price']:,.2f}")
        st.metric("24h Change", f"{latest['change_24h']:+.2f}%")
        st.metric("Data Source", latest['source'].upper())
        st.metric("Total Updates", len(st.session_state.btc_prices))
        st.metric("Messages Received", st.session_state.messages_received)
    else:
        st.info("No data yet")
    
    st.markdown("---")
    st.markdown("### 🔄 How It Works")
    
    st.markdown("""
    **1. Publisher (Celery Task)**
    - Fetches BTC price every 5 seconds
    - Publishes to Redis channel: `btc_price_updates`
    
    **2. Subscriber (This Dashboard)**
    - Subscribes to `btc_price_updates`
    - Receives messages instantly
    - Updates UI in real-time
    
    **3. Benefits**
    - ✅ Real-time updates (no polling)
    - ✅ Low latency
    - ✅ Scalable (many subscribers)
    - ✅ Decoupled architecture
    """)

# Message log
st.markdown("---")
st.markdown("### 📝 Recent Messages")

if st.session_state.btc_prices:
    # Show last 10 messages
    recent = st.session_state.btc_prices[-10:][::-1]
    
    for msg in recent:
        with st.expander(f"${msg['price']:,.2f} - {msg['timestamp'].strftime('%H:%M:%S')}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Price", f"${msg['price']:,.2f}")
            with col2:
                st.metric("Change 24h", f"{msg['change_24h']:+.2f}%")
            with col3:
                st.metric("Source", msg['source'].upper())
else:
    st.info("No messages received yet. Make sure the Celery task is running and publishing to 'btc_price_updates' channel.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔄 This dashboard demonstrates Redis Pub/Sub for real-time BTC price updates</p>
    <p>📡 Channel: <code>btc_price_updates</code> | Publisher: Celery Task (every 5s)</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh when subscribed
if st.session_state.subscriber_running:
    time.sleep(1)
    st.rerun()

