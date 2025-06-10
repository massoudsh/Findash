#!/usr/bin/env python3
"""
Example usage of Quantum Trading Matrix™ with Direct PostgreSQL
This demonstrates the new PostgreSQL repository pattern instead of SQLAlchemy ORM
"""

import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import database modules
from database.postgres_connection import get_db
from database.repositories import (
    UserRepository, PortfolioRepository, OptionPositionRepository,
    MarketDataRepository, APIKeyRepository, AuditLogRepository
)

# Import options trading modules
from options_risk_integration import (
    BlackScholesCalculator, OptionsPortfolio, OptionPosition,
    CorrelationAnalyzer, RiskReportGenerator
)

def main():
    """Demonstrate PostgreSQL-based trading system"""
    print("🚀 Quantum Trading Matrix™ - PostgreSQL Example")
    print("=" * 50)
    
    # Initialize repositories
    user_repo = UserRepository()
    portfolio_repo = PortfolioRepository()
    position_repo = OptionPositionRepository()
    market_data_repo = MarketDataRepository()
    audit_repo = AuditLogRepository()
    
    try:
        # Test database connection
        db = get_db()
        db.execute_query("SELECT version()", fetch='one')
        print("✅ PostgreSQL connection successful")
        
        # 1. User Management
        print("\n📋 1. User Management Example")
        print("-" * 30)
        
        # Create or get demo user
        demo_user = user_repo.get_user_by_email("demo@quantumtrading.com")
        if not demo_user:
            print("👤 Creating demo user...")
            demo_user = user_repo.create_user(
                email="demo@quantumtrading.com",
                username="demo_user",
                password="demo123",
                full_name="Demo User"
            )
        
        print(f"User: {demo_user.email} (ID: {demo_user.id})")
        print(f"Risk Tolerance: {demo_user.risk_tolerance}")
        
        # 2. Portfolio Management
        print("\n💼 2. Portfolio Management Example")
        print("-" * 35)
        
        # Get or create portfolio
        portfolios = portfolio_repo.get_user_portfolios(demo_user.id)
        if portfolios:
            portfolio = portfolios[0]
            print(f"Using existing portfolio: {portfolio.name}")
        else:
            portfolio = portfolio_repo.create_portfolio(
                user_id=demo_user.id,
                name="Demo Options Portfolio",
                description="Portfolio for testing PostgreSQL integration",
                initial_capital=100000.0
            )
            print(f"Created new portfolio: {portfolio.name}")
        
        print(f"Portfolio Value: ${portfolio.current_value:,.2f}")
        print(f"Cash Balance: ${portfolio.cash_balance:,.2f}")
        
        # 3. Options Trading
        print("\n📈 3. Options Trading Example")
        print("-" * 30)
        
        # Create sample option position
        expiry_date = datetime.now() + timedelta(days=30)
        underlying_price = 155.0
        strike_price = 150.0
        volatility = 0.25
        
        # Calculate option price using Black-Scholes
        time_to_expiry = 30 / 365.0
        option_price = BlackScholesCalculator.option_price(
            underlying_price, strike_price, time_to_expiry,
            0.05, volatility, 'call'
        )
        
        print(f"AAPL Call Option Analysis:")
        print(f"  Underlying Price: ${underlying_price}")
        print(f"  Strike Price: ${strike_price}")
        print(f"  Time to Expiry: {30} days")
        print(f"  Calculated Option Price: ${option_price:.4f}")
        
        # Create position in database
        position = position_repo.create_position(
            user_id=demo_user.id,
            portfolio_id=portfolio.id,
            symbol="AAPL",
            option_type="call",
            strike_price=strike_price,
            expiry_date=expiry_date,
            quantity=10,
            premium_paid=option_price,
            underlying_price=underlying_price,
            implied_volatility=volatility
        )
        
        print(f"✅ Created option position: {position.id}")
        
        # Calculate and update Greeks
        greeks = BlackScholesCalculator.calculate_greeks(
            underlying_price, strike_price, time_to_expiry,
            0.05, volatility, 'call'
        )
        
        position_repo.update_position_greeks(
            position.id,
            greeks.delta, greeks.gamma, greeks.theta,
            greeks.vega, greeks.rho, option_price
        )
        
        print(f"Greeks Updated:")
        print(f"  Delta: {greeks.delta:.4f}")
        print(f"  Gamma: {greeks.gamma:.4f}")
        print(f"  Theta: {greeks.theta:.4f}")
        print(f"  Vega: {greeks.vega:.4f}")
        print(f"  Rho: {greeks.rho:.4f}")
        
        # 4. Portfolio Analysis
        print("\n📊 4. Portfolio Risk Analysis")
        print("-" * 30)
        
        # Get all user positions
        positions = position_repo.get_user_positions(demo_user.id)
        print(f"Total Positions: {len(positions)}")
        
        # Calculate portfolio-level Greeks
        if positions:
            total_delta = sum(pos.delta * pos.quantity for pos in positions if pos.delta)
            total_gamma = sum(pos.gamma * pos.quantity for pos in positions if pos.gamma)
            total_theta = sum(pos.theta * pos.quantity for pos in positions if pos.theta)
            
            print(f"Portfolio Greeks:")
            print(f"  Total Delta: {total_delta:.4f}")
            print(f"  Total Gamma: {total_gamma:.4f}")
            print(f"  Total Theta: {total_theta:.4f}")
            
            # Generate risk report using OptionsPortfolio
            options_portfolio = OptionsPortfolio()
            for pos in positions:
                opt_pos = OptionPosition(
                    symbol=pos.symbol,
                    option_type=pos.option_type,
                    strike=pos.strike_price,
                    expiry=pos.expiry_date,
                    quantity=pos.quantity,
                    premium=pos.premium_paid,
                    underlying_price=pos.underlying_price,
                    volatility=pos.implied_volatility
                )
                options_portfolio.add_position(opt_pos)
            
            # Risk analysis
            correlation_analyzer = CorrelationAnalyzer()
            report_generator = RiskReportGenerator(options_portfolio, correlation_analyzer)
            risk_dashboard = report_generator.generate_risk_dashboard()
            
            print(f"\nRisk Analysis:")
            portfolio_summary = risk_dashboard['portfolio_summary']
            print(f"  Portfolio Value: ${portfolio_summary['portfolio_value']:,.2f}")
            print(f"  VaR (95%): ${portfolio_summary['portfolio_var_95']:,.2f}")
            
            # Show risk alerts
            alerts = risk_dashboard['risk_alerts']
            if alerts:
                print(f"\n⚠️  Risk Alerts:")
                for alert in alerts:
                    print(f"  - {alert['type']}: {alert['message']}")
            else:
                print(f"✅ No risk alerts")
        
        # 5. Market Data
        print("\n📈 5. Market Data Example")
        print("-" * 25)
        
        # Save market data
        market_data_id = market_data_repo.save_market_data(
            symbol="AAPL",
            open_price=154.0,
            high_price=157.0,
            low_price=153.0,
            close_price=underlying_price,
            volume=1000000,
            timestamp=datetime.now(),
            implied_volatility=volatility
        )
        
        print(f"✅ Saved market data: {market_data_id}")
        
        # Get latest price
        latest_price = market_data_repo.get_latest_price("AAPL")
        if latest_price:
            print(f"Latest AAPL price: ${latest_price['close_price']}")
        
        # 6. Audit Logging
        print("\n📝 6. Audit Logging Example")
        print("-" * 27)
        
        # Log an action
        audit_id = audit_repo.log_action(
            user_id=demo_user.id,
            action="create_option_position",
            resource_type="option_position",
            resource_id=position.id,
            new_values={
                "symbol": "AAPL",
                "option_type": "call",
                "quantity": 10
            },
            ip_address="127.0.0.1"
        )
        
        print(f"✅ Logged action: {audit_id}")
        
        print("\n🎉 PostgreSQL Example completed successfully!")
        print("\nKey Benefits of Direct PostgreSQL:")
        print("✅ No ORM overhead - direct SQL queries")
        print("✅ Better performance and control")
        print("✅ Easier to optimize and debug")
        print("✅ Native PostgreSQL features (JSONB, triggers, etc.)")
        print("✅ Connection pooling for scalability")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

def cleanup_demo_data():
    """Clean up demo data (optional)"""
    print("\n🧹 Cleaning up demo data...")
    
    user_repo = UserRepository()
    position_repo = OptionPositionRepository()
    
    # Get demo user
    demo_user = user_repo.get_user_by_email("demo@quantumtrading.com")
    if demo_user:
        # Delete all positions for demo user
        positions = position_repo.get_user_positions(demo_user.id)
        for position in positions:
            position_repo.delete_position(position.id)
        
        print(f"✅ Cleaned up {len(positions)} positions")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PostgreSQL Trading System Example')
    parser.add_argument('--cleanup', action='store_true', 
                       help='Clean up demo data after running')
    
    args = parser.parse_args()
    
    # Run the main example
    result = main()
    
    # Clean up if requested
    if args.cleanup:
        cleanup_demo_data()
    
    sys.exit(result) 