"""
Regulatory Compliance Engine
Advanced compliance monitoring and enforcement

This module handles:
- Pattern Day Trader (PDT) detection and controls
- Position limits and concentration controls
- Wash sale rule enforcement
- Best execution monitoring
- Trade reporting and audit trails
- Regulatory risk controls
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class ComplianceRuleType(Enum):
    PATTERN_DAY_TRADER = "pattern_day_trader"
    POSITION_LIMIT = "position_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    WASH_SALE = "wash_sale"
    BEST_EXECUTION = "best_execution"
    MARGIN_REQUIREMENT = "margin_requirement"
    SHORT_SALE = "short_sale"

class ViolationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"

class ComplianceAction(Enum):
    ALLOW = "allow"
    WARN = "warn"
    RESTRICT = "restrict"
    BLOCK = "block"
    REPORT = "report"

@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    violation_id: str
    rule_type: ComplianceRuleType
    severity: ViolationSeverity
    description: str
    account_id: str
    symbol: Optional[str] = None
    quantity: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Additional context
    context: Dict[str, Any] = field(default_factory=dict)
    action_taken: Optional[ComplianceAction] = None
    resolved: bool = False

@dataclass
class TradingLimit:
    """Trading limit configuration"""
    limit_type: str
    symbol: Optional[str] = None
    max_quantity: Optional[float] = None
    max_value: Optional[float] = None
    max_percentage: Optional[float] = None  # of portfolio
    time_window: Optional[timedelta] = None
    enabled: bool = True

@dataclass
class PDTStatus:
    """Pattern Day Trader status"""
    is_pdt: bool
    day_trades_count: int
    day_trades_remaining: int
    day_trading_buying_power: float
    restrictions: List[str] = field(default_factory=list)

class PatternDayTraderMonitor:
    """Pattern Day Trader compliance monitoring"""
    
    def __init__(self):
        self.day_trades_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.pdt_threshold = 25000.0  # $25k minimum equity
        self.max_day_trades = 3  # In 5 business days
    
    async def check_day_trade_count(
        self,
        account_id: str,
        trades: List[Dict[str, Any]],
        lookback_days: int = 5
    ) -> int:
        """Count day trades in lookback period"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
        day_trades = []
        
        # Group trades by symbol and day
        symbol_trades = {}
        for trade in trades:
            if trade['timestamp'] < cutoff_date:
                continue
                
            symbol = trade['symbol']
            trade_date = trade['timestamp'].date()
            
            if symbol not in symbol_trades:
                symbol_trades[symbol] = {}
            
            if trade_date not in symbol_trades[symbol]:
                symbol_trades[symbol][trade_date] = []
            
            symbol_trades[symbol][trade_date].append(trade)
        
        # Identify day trades (buy and sell same symbol same day)
        for symbol, date_trades in symbol_trades.items():
            for trade_date, trades_on_date in date_trades.items():
                buys = [t for t in trades_on_date if t['side'] == 'buy']
                sells = [t for t in trades_on_date if t['side'] == 'sell']
                
                if buys and sells:
                    # This constitutes a day trade
                    day_trades.append({
                        'symbol': symbol,
                        'date': trade_date,
                        'buy_quantity': sum(t['quantity'] for t in buys),
                        'sell_quantity': sum(t['quantity'] for t in sells)
                    })
        
        return len(day_trades)
    
    async def get_pdt_status(
        self,
        account_id: str,
        account_value: float,
        recent_trades: List[Dict[str, Any]]
    ) -> PDTStatus:
        """Get Pattern Day Trader status"""
        
        day_trades_count = await self.check_day_trade_count(account_id, recent_trades)
        
        # Check if flagged as PDT
        is_pdt = day_trades_count >= 4 or account_value >= self.pdt_threshold
        
        # Calculate remaining day trades
        if is_pdt:
            day_trades_remaining = float('inf')  # No limit for PDT with sufficient equity
        else:
            day_trades_remaining = max(0, self.max_day_trades - day_trades_count)
        
        # Calculate day trading buying power
        if is_pdt and account_value >= self.pdt_threshold:
            day_trading_buying_power = account_value * 4  # 4:1 leverage
        else:
            day_trading_buying_power = account_value  # 1:1 for non-PDT
        
        # Determine restrictions
        restrictions = []
        if is_pdt and account_value < self.pdt_threshold:
            restrictions.append("PDT account below $25k minimum - day trading restricted")
        elif not is_pdt and day_trades_remaining == 0:
            restrictions.append("Maximum day trades reached - further day trading restricted")
        
        return PDTStatus(
            is_pdt=is_pdt,
            day_trades_count=day_trades_count,
            day_trades_remaining=day_trades_remaining,
            day_trading_buying_power=day_trading_buying_power,
            restrictions=restrictions
        )

class WashSaleMonitor:
    """Wash sale rule compliance monitoring"""
    
    def __init__(self):
        self.wash_sale_window = timedelta(days=30)  # 30 days before and after
    
    async def check_wash_sale(
        self,
        symbol: str,
        sale_date: datetime,
        sale_quantity: float,
        sale_price: float,
        trades: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Check for wash sale violation"""
        
        # Check if the sale resulted in a loss
        # Find the corresponding purchase(s)
        purchases = [
            t for t in trades 
            if (t['symbol'] == symbol and 
                t['side'] == 'buy' and 
                t['timestamp'] <= sale_date)
        ]
        
        if not purchases:
            return None
        
        # Calculate if there's a loss (simplified - FIFO assumption)
        avg_purchase_price = np.mean([t['price'] for t in purchases])
        
        if sale_price >= avg_purchase_price:
            return None  # No loss, no wash sale concern
        
        # Check for purchases within wash sale window after the sale
        wash_sale_start = sale_date
        wash_sale_end = sale_date + self.wash_sale_window
        
        subsequent_purchases = [
            t for t in trades
            if (t['symbol'] == symbol and
                t['side'] == 'buy' and
                wash_sale_start < t['timestamp'] <= wash_sale_end)
        ]
        
        if subsequent_purchases:
            total_subsequent_quantity = sum(t['quantity'] for t in subsequent_purchases)
            
            # Wash sale if any subsequent purchase within 30 days
            wash_quantity = min(sale_quantity, total_subsequent_quantity)
            disallowed_loss = (avg_purchase_price - sale_price) * wash_quantity
            
            return {
                'is_wash_sale': True,
                'wash_quantity': wash_quantity,
                'disallowed_loss': disallowed_loss,
                'subsequent_purchases': len(subsequent_purchases),
                'message': f"Wash sale detected: ${disallowed_loss:.2f} loss disallowed"
            }
        
        return None

class PositionLimitMonitor:
    """Position and concentration limit monitoring"""
    
    def __init__(self):
        self.default_limits = {
            'max_position_percentage': 0.10,  # 10% of portfolio max per position
            'max_sector_percentage': 0.25,    # 25% max per sector
            'max_single_order_value': 100000  # $100k max single order
        }
    
    async def check_position_limits(
        self,
        symbol: str,
        quantity: float,
        price: float,
        current_positions: Dict[str, Dict[str, Any]],
        portfolio_value: float,
        custom_limits: Optional[Dict[str, TradingLimit]] = None
    ) -> List[ComplianceViolation]:
        """Check position and concentration limits"""
        
        violations = []
        order_value = quantity * price
        
        # Check single order value limit
        max_order_value = self.default_limits['max_single_order_value']
        if custom_limits and 'max_order_value' in custom_limits:
            max_order_value = custom_limits['max_order_value'].max_value
        
        if order_value > max_order_value:
            violations.append(ComplianceViolation(
                violation_id=f"order_limit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                rule_type=ComplianceRuleType.POSITION_LIMIT,
                severity=ViolationSeverity.MAJOR,
                description=f"Order value ${order_value:,.2f} exceeds limit ${max_order_value:,.2f}",
                account_id="",  # To be filled by caller
                symbol=symbol,
                quantity=quantity,
                context={"order_value": order_value, "limit": max_order_value}
            ))
        
        # Check position concentration limit
        current_position_value = current_positions.get(symbol, {}).get('market_value', 0)
        new_position_value = current_position_value + order_value
        position_percentage = new_position_value / portfolio_value if portfolio_value > 0 else 0
        
        max_position_pct = self.default_limits['max_position_percentage']
        if custom_limits and 'max_position_percentage' in custom_limits:
            max_position_pct = custom_limits['max_position_percentage'].max_percentage
        
        if position_percentage > max_position_pct:
            violations.append(ComplianceViolation(
                violation_id=f"concentration_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                rule_type=ComplianceRuleType.CONCENTRATION_LIMIT,
                severity=ViolationSeverity.WARNING,
                description=f"Position in {symbol} would be {position_percentage:.1%} of portfolio, exceeding {max_position_pct:.1%} limit",
                account_id="",
                symbol=symbol,
                quantity=quantity,
                context={
                    "position_percentage": position_percentage,
                    "limit_percentage": max_position_pct,
                    "new_position_value": new_position_value
                }
            ))
        
        return violations

class BestExecutionMonitor:
    """Best execution compliance monitoring"""
    
    def __init__(self):
        self.execution_benchmarks = {
            'max_slippage_bps': 10,  # 10 basis points
            'max_market_impact_bps': 25,  # 25 basis points
            'min_fill_rate': 0.95  # 95% fill rate
        }
    
    async def analyze_execution_quality(
        self,
        orders: List[Dict[str, Any]],
        market_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze execution quality against benchmarks"""
        
        if not orders:
            return {"message": "No orders to analyze"}
        
        execution_metrics = {
            'total_orders': len(orders),
            'filled_orders': 0,
            'total_slippage_bps': 0,
            'total_market_impact_bps': 0,
            'violations': []
        }
        
        for order in orders:
            if order.get('status') != 'filled':
                continue
            
            execution_metrics['filled_orders'] += 1
            symbol = order['symbol']
            fill_price = order.get('avg_fill_price')
            
            if not fill_price or symbol not in market_data:
                continue
            
            # Calculate slippage
            mid_price = (market_data[symbol]['bid'] + market_data[symbol]['ask']) / 2
            slippage = abs(fill_price - mid_price) / mid_price * 10000  # bps
            
            execution_metrics['total_slippage_bps'] += slippage
            
            # Check slippage violation
            if slippage > self.execution_benchmarks['max_slippage_bps']:
                execution_metrics['violations'].append({
                    'type': 'excessive_slippage',
                    'order_id': order.get('order_id'),
                    'symbol': symbol,
                    'slippage_bps': slippage,
                    'benchmark_bps': self.execution_benchmarks['max_slippage_bps']
                })
        
        # Calculate averages
        filled_count = execution_metrics['filled_orders']
        if filled_count > 0:
            execution_metrics['avg_slippage_bps'] = execution_metrics['total_slippage_bps'] / filled_count
            execution_metrics['fill_rate'] = filled_count / execution_metrics['total_orders']
        else:
            execution_metrics['avg_slippage_bps'] = 0
            execution_metrics['fill_rate'] = 0
        
        # Check fill rate violation
        if execution_metrics['fill_rate'] < self.execution_benchmarks['min_fill_rate']:
            execution_metrics['violations'].append({
                'type': 'low_fill_rate',
                'fill_rate': execution_metrics['fill_rate'],
                'benchmark': self.execution_benchmarks['min_fill_rate']
            })
        
        return execution_metrics

class RegulatoryComplianceEngine:
    """Main regulatory compliance engine"""
    
    def __init__(self):
        self.pdt_monitor = PatternDayTraderMonitor()
        self.wash_sale_monitor = WashSaleMonitor()
        self.position_limit_monitor = PositionLimitMonitor()
        self.best_execution_monitor = BestExecutionMonitor()
        
        # Compliance state
        self.violations: List[ComplianceViolation] = []
        self.trading_limits: Dict[str, TradingLimit] = {}
        self.restricted_accounts: Set[str] = set()
        
        # Audit trail
        self.audit_log: List[Dict[str, Any]] = []
    
    async def pre_trade_compliance_check(
        self,
        account_id: str,
        symbol: str,
        quantity: float,
        side: str,
        price: float,
        order_type: str,
        account_info: Dict[str, Any],
        current_positions: Dict[str, Dict[str, Any]],
        recent_trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Comprehensive pre-trade compliance check"""
        
        compliance_result = {
            'approved': True,
            'action': ComplianceAction.ALLOW,
            'violations': [],
            'warnings': [],
            'restrictions': []
        }
        
        # Check if account is restricted
        if account_id in self.restricted_accounts:
            compliance_result['approved'] = False
            compliance_result['action'] = ComplianceAction.BLOCK
            compliance_result['restrictions'].append("Account is currently restricted")
            return compliance_result
        
        # PDT compliance check
        try:
            pdt_status = await self.pdt_monitor.get_pdt_status(
                account_id, 
                account_info.get('total_value', 0),
                recent_trades
            )
            
            # Check if this would be a day trade
            is_potential_day_trade = await self._is_potential_day_trade(
                symbol, side, recent_trades
            )
            
            if is_potential_day_trade and pdt_status.day_trades_remaining == 0:
                violation = ComplianceViolation(
                    violation_id=f"pdt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    rule_type=ComplianceRuleType.PATTERN_DAY_TRADER,
                    severity=ViolationSeverity.MAJOR,
                    description="Day trade would exceed PDT limits",
                    account_id=account_id,
                    symbol=symbol,
                    quantity=quantity,
                    context={"pdt_status": pdt_status.__dict__}
                )
                
                compliance_result['violations'].append(violation)
                compliance_result['approved'] = False
                compliance_result['action'] = ComplianceAction.BLOCK
                
        except Exception as e:
            logger.error(f"PDT check failed: {e}")
            compliance_result['warnings'].append(f"PDT check failed: {e}")
        
        # Position limit checks
        try:
            limit_violations = await self.position_limit_monitor.check_position_limits(
                symbol, quantity, price, current_positions, 
                account_info.get('total_value', 0), self.trading_limits
            )
            
            for violation in limit_violations:
                violation.account_id = account_id
                
                if violation.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.MAJOR]:
                    compliance_result['violations'].append(violation)
                    compliance_result['approved'] = False
                    compliance_result['action'] = ComplianceAction.BLOCK
                else:
                    compliance_result['warnings'].append(violation)
                    if compliance_result['action'] == ComplianceAction.ALLOW:
                        compliance_result['action'] = ComplianceAction.WARN
        
        except Exception as e:
            logger.error(f"Position limit check failed: {e}")
            compliance_result['warnings'].append(f"Position limit check failed: {e}")
        
        # Wash sale check for sell orders
        if side.lower() == 'sell':
            try:
                wash_sale_result = await self.wash_sale_monitor.check_wash_sale(
                    symbol, datetime.utcnow(), quantity, price, recent_trades
                )
                
                if wash_sale_result and wash_sale_result.get('is_wash_sale'):
                    violation = ComplianceViolation(
                        violation_id=f"wash_sale_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                        rule_type=ComplianceRuleType.WASH_SALE,
                        severity=ViolationSeverity.WARNING,
                        description=wash_sale_result['message'],
                        account_id=account_id,
                        symbol=symbol,
                        quantity=quantity,
                        context=wash_sale_result
                    )
                    
                    compliance_result['warnings'].append(violation)
                    if compliance_result['action'] == ComplianceAction.ALLOW:
                        compliance_result['action'] = ComplianceAction.WARN
                        
            except Exception as e:
                logger.error(f"Wash sale check failed: {e}")
                compliance_result['warnings'].append(f"Wash sale check failed: {e}")
        
        # Log compliance check
        self._log_compliance_check(account_id, symbol, quantity, side, compliance_result)
        
        return compliance_result
    
    async def post_trade_compliance_review(
        self,
        trade: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Post-trade compliance review"""
        
        review_result = {
            'trade_id': trade.get('trade_id'),
            'compliance_score': 1.0,  # 0.0 to 1.0
            'issues': [],
            'recommendations': []
        }
        
        # Best execution analysis
        try:
            execution_analysis = await self.best_execution_monitor.analyze_execution_quality(
                [trade], {trade['symbol']: market_data}
            )
            
            if execution_analysis.get('violations'):
                for violation in execution_analysis['violations']:
                    review_result['issues'].append(violation)
                    review_result['compliance_score'] *= 0.9  # Reduce score
        
        except Exception as e:
            logger.error(f"Post-trade execution analysis failed: {e}")
            review_result['issues'].append(f"Execution analysis failed: {e}")
        
        return review_result
    
    async def _is_potential_day_trade(
        self,
        symbol: str,
        side: str,
        recent_trades: List[Dict[str, Any]]
    ) -> bool:
        """Check if order would constitute a day trade"""
        
        today = datetime.utcnow().date()
        
        # Check for opposite side trades today
        opposite_side = 'sell' if side == 'buy' else 'buy'
        
        today_trades = [
            t for t in recent_trades
            if (t['symbol'] == symbol and 
                t['side'] == opposite_side and
                t['timestamp'].date() == today)
        ]
        
        return len(today_trades) > 0
    
    def _log_compliance_check(
        self,
        account_id: str,
        symbol: str,
        quantity: float,
        side: str,
        result: Dict[str, Any]
    ):
        """Log compliance check to audit trail"""
        
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'pre_trade_compliance_check',
            'account_id': account_id,
            'symbol': symbol,
            'quantity': quantity,
            'side': side,
            'approved': result['approved'],
            'action': result['action'].value,
            'violation_count': len(result['violations']),
            'warning_count': len(result['warnings'])
        }
        
        self.audit_log.append(audit_entry)
        
        # Keep only last 10000 entries
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]
    
    def add_trading_limit(self, limit: TradingLimit):
        """Add or update trading limit"""
        
        limit_key = f"{limit.limit_type}_{limit.symbol or 'global'}"
        self.trading_limits[limit_key] = limit
        
        logger.info(f"Added trading limit: {limit_key}")
    
    def restrict_account(self, account_id: str, reason: str):
        """Restrict trading account"""
        
        self.restricted_accounts.add(account_id)
        
        # Log restriction
        self.audit_log.append({
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'account_restriction',
            'account_id': account_id,
            'reason': reason
        })
        
        logger.warning(f"Account {account_id} restricted: {reason}")
    
    def unrestrict_account(self, account_id: str):
        """Remove account restriction"""
        
        if account_id in self.restricted_accounts:
            self.restricted_accounts.remove(account_id)
            
            # Log unrestriction
            self.audit_log.append({
                'timestamp': datetime.utcnow().isoformat(),
                'type': 'account_unrestriction',
                'account_id': account_id
            })
            
            logger.info(f"Account {account_id} unrestricted")
    
    def get_compliance_report(
        self,
        account_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate compliance report"""
        
        # Filter violations
        filtered_violations = self.violations
        
        if account_id:
            filtered_violations = [v for v in filtered_violations if v.account_id == account_id]
        
        if start_date:
            filtered_violations = [v for v in filtered_violations if v.timestamp >= start_date]
        
        if end_date:
            filtered_violations = [v for v in filtered_violations if v.timestamp <= end_date]
        
        # Aggregate statistics
        violation_by_type = {}
        violation_by_severity = {}
        
        for violation in filtered_violations:
            rule_type = violation.rule_type.value
            severity = violation.severity.value
            
            violation_by_type[rule_type] = violation_by_type.get(rule_type, 0) + 1
            violation_by_severity[severity] = violation_by_severity.get(severity, 0) + 1
        
        return {
            'report_period': {
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None
            },
            'summary': {
                'total_violations': len(filtered_violations),
                'violation_by_type': violation_by_type,
                'violation_by_severity': violation_by_severity,
                'restricted_accounts': len(self.restricted_accounts)
            },
            'violations': [
                {
                    'violation_id': v.violation_id,
                    'rule_type': v.rule_type.value,
                    'severity': v.severity.value,
                    'description': v.description,
                    'account_id': v.account_id,
                    'symbol': v.symbol,
                    'timestamp': v.timestamp.isoformat(),
                    'resolved': v.resolved
                }
                for v in filtered_violations
            ],
            'trading_limits': {
                key: {
                    'limit_type': limit.limit_type,
                    'symbol': limit.symbol,
                    'max_quantity': limit.max_quantity,
                    'max_value': limit.max_value,
                    'max_percentage': limit.max_percentage,
                    'enabled': limit.enabled
                }
                for key, limit in self.trading_limits.items()
            }
        } 