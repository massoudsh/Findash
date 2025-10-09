from sqlalchemy.orm import Session
from .models import User, Portfolio, Position, Trade, PortfolioSnapshot, MarketData, NewsArticle, RedditSentiment, AlertRule
from src.schemas import UserCreate, PortfolioCreate, PositionCreate, TradeCreate, PortfolioSnapshotCreate
from typing import List, Optional, Dict
from sqlalchemy.exc import NoResultFound
from datetime import datetime

# --- User CRUD ---
def create_user(db: Session, user: UserCreate) -> User:
    db_user = User(
        username=user.username,
        email=user.email,
        password_hash=user.password  # Hash in real app!
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()

# --- Portfolio CRUD ---
def create_portfolio(db: Session, user_id: int, portfolio: PortfolioCreate) -> Portfolio:
    db_portfolio = Portfolio(
        user_id=user_id,
        name=portfolio.name,
        description=portfolio.description
    )
    db.add(db_portfolio)
    db.commit()
    db.refresh(db_portfolio)
    return db_portfolio

def get_portfolios_by_user(db: Session, user_id: int) -> List[Portfolio]:
    return db.query(Portfolio).filter(Portfolio.user_id == user_id).all()

def get_portfolio(db: Session, portfolio_id: int) -> Optional[Portfolio]:
    return db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()

def get_all_portfolios(db: Session) -> List[Portfolio]:
    return db.query(Portfolio).all()

def delete_portfolio(db: Session, portfolio_id: int):
    db_portfolio = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if db_portfolio:
        db.delete(db_portfolio)
        db.commit()

# --- Position CRUD ---
def create_position(db: Session, portfolio_id: int, position: PositionCreate) -> Position:
    db_position = Position(
        portfolio_id=portfolio_id,
        symbol=position.symbol,
        quantity=position.quantity,
        average_price=position.average_price
    )
    db.add(db_position)
    db.commit()
    db.refresh(db_position)
    return db_position

def get_positions_by_portfolio(db: Session, portfolio_id: int) -> List[Position]:
    return db.query(Position).filter(Position.portfolio_id == portfolio_id).all()

def get_position(db: Session, position_id: int) -> Optional[Position]:
    return db.query(Position).filter(Position.id == position_id).first()

def delete_position(db: Session, position_id: int):
    db_position = db.query(Position).filter(Position.id == position_id).first()
    if db_position:
        db.delete(db_position)
        db.commit()

# --- Trade CRUD ---
def create_trade(db: Session, portfolio_id: int, trade: TradeCreate) -> Trade:
    db_trade = Trade(
        portfolio_id=portfolio_id,
        symbol=trade.symbol,
        trade_type=trade.trade_type,
        quantity=trade.quantity,
        price=trade.price,
        notes=trade.notes
    )
    db.add(db_trade)
    db.commit()
    db.refresh(db_trade)
    return db_trade

def get_trades_by_portfolio(db: Session, portfolio_id: int) -> List[Trade]:
    return db.query(Trade).filter(Trade.portfolio_id == portfolio_id).all()

def get_trade(db: Session, trade_id: int) -> Optional[Trade]:
    return db.query(Trade).filter(Trade.id == trade_id).first()

def delete_trade(db: Session, trade_id: int):
    db_trade = db.query(Trade).filter(Trade.id == trade_id).first()
    if db_trade:
        db.delete(db_trade)
        db.commit()

# --- PortfolioSnapshot CRUD ---
def create_portfolio_snapshot(db: Session, portfolio_id: int, snapshot: PortfolioSnapshotCreate) -> PortfolioSnapshot:
    db_snapshot = PortfolioSnapshot(
        portfolio_id=portfolio_id,
        snapshot_time=snapshot.snapshot_time,
        total_value=snapshot.total_value
    )
    db.add(db_snapshot)
    db.commit()
    db.refresh(db_snapshot)
    return db_snapshot

def get_snapshots_by_portfolio(db: Session, portfolio_id: int) -> List[PortfolioSnapshot]:
    return db.query(PortfolioSnapshot).filter(PortfolioSnapshot.portfolio_id == portfolio_id).all()

def get_snapshot(db: Session, snapshot_id: int) -> Optional[PortfolioSnapshot]:
    return db.query(PortfolioSnapshot).filter(PortfolioSnapshot.id == snapshot_id).first()

def delete_snapshot(db: Session, snapshot_id: int):
    db_snapshot = db.query(PortfolioSnapshot).filter(PortfolioSnapshot.id == snapshot_id).first()
    if db_snapshot:
        db.delete(db_snapshot)
        db.commit()

def create_market_data(db: Session, data: dict) -> MarketData:
    db_data = MarketData(
        time=data['timestamp'],
        symbol=data['symbol'],
        price=data.get('price'),
        open=data.get('open'),
        high=data.get('high'),
        low=data.get('low'),
        volume=data.get('volume'),
        exchange=data.get('exchange')
    )
    db.add(db_data)
    db.commit()
    db.refresh(db_data)
    return db_data

def create_news_article(db: Session, article: dict) -> NewsArticle:
    db_article = NewsArticle(
        title=article['title'],
        summary=article.get('summary'),
        url=article.get('url'),
        published_at=article.get('published_at', datetime.utcnow()),
        source=article.get('source', 'Finextra'),
        symbol=article.get('symbol')
    )
    db.add(db_article)
    db.commit()
    db.refresh(db_article)
    return db_article

def create_reddit_sentiment(db: Session, sentiment: dict) -> RedditSentiment:
    db_sentiment = RedditSentiment(
        subreddit=sentiment['subreddit'],
        positive=sentiment.get('positive'),
        negative=sentiment.get('negative'),
        neutral=sentiment.get('neutral'),
        top_keywords=','.join(sentiment.get('top_keywords', [])),
        sample_size=sentiment.get('sample_size'),
        analyzed_at=sentiment.get('analyzed_at', datetime.utcnow())
    )
    db.add(db_sentiment)
    db.commit()
    db.refresh(db_sentiment)
    return db_sentiment

# --- AlertRule CRUD ---
def create_alert_rule(db: Session, user_id: int, data: Dict) -> AlertRule:
    alert_rule = AlertRule(
        user_id=user_id,
        name=data['name'],
        description=data.get('description'),
        category=data['category'],
        metric=data['metric'],
        operator=data['operator'],
        threshold=data['threshold'],
        duration=data['duration'],
        notifications=data.get('notifications', {}),
        severity=data.get('severity', 'medium'),
        enabled=data.get('enabled', True)
    )
    db.add(alert_rule)
    db.commit()
    db.refresh(alert_rule)
    return alert_rule

def get_alert_rules_by_user(db: Session, user_id: int) -> List[AlertRule]:
    return db.query(AlertRule).filter(AlertRule.user_id == user_id).all()

def get_alert_rule(db: Session, alert_rule_id: int) -> Optional[AlertRule]:
    return db.query(AlertRule).filter(AlertRule.id == alert_rule_id).first()

def update_alert_rule(db: Session, alert_rule_id: int, data: Dict) -> Optional[AlertRule]:
    alert_rule = db.query(AlertRule).filter(AlertRule.id == alert_rule_id).first()
    if not alert_rule:
        return None
    for key, value in data.items():
        if hasattr(alert_rule, key):
            setattr(alert_rule, key, value)
    db.commit()
    db.refresh(alert_rule)
    return alert_rule

def delete_alert_rule(db: Session, alert_rule_id: int) -> bool:
    alert_rule = db.query(AlertRule).filter(AlertRule.id == alert_rule_id).first()
    if alert_rule:
        db.delete(alert_rule)
        db.commit()
        return True
    return False 