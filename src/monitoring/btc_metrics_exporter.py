#!/usr/bin/env python3
"""
BTC Price Metrics Exporter
Exposes BTC API metrics for Prometheus scraping
"""

import os
from prometheus_client import start_http_server, REGISTRY
from src.data_processing.btc_price_tracker import (
    btc_api_calls_total,
    btc_api_latency_seconds,
    btc_price_current,
    btc_price_change_24h,
    btc_api_cache_hits,
    btc_api_cache_misses
)

def main():
    """Start metrics server"""
    port = int(os.getenv('METRICS_PORT', '8003'))
    host = os.getenv('METRICS_HOST', '0.0.0.0')
    
    print(f"Starting BTC metrics server on {host}:{port}")
    start_http_server(port, addr=host)
    
    print(f"BTC metrics available at http://localhost:{port}/metrics")
    print("Metrics exposed:")
    print("  - btc_api_calls_total")
    print("  - btc_api_latency_seconds")
    print("  - btc_price_current_usd")
    print("  - btc_price_change_24h_percent")
    print("  - btc_api_cache_hits_total")
    print("  - btc_api_cache_misses_total")
    
    # Keep server running
    import time
    while True:
        time.sleep(1)

if __name__ == '__main__':
    main()

