"""
Octopus Trading Platform™ - Audit & Compliance Service
Enterprise-grade audit trail and regulatory compliance
"""

import logging
import json
import hashlib
import hmac
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from decimal import Decimal
import asyncio
import asyncpg
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)

# Compliance Standards
class ComplianceStandard(str, Enum):
    MIFID_II = "mifid_ii"  # Markets in Financial Instruments Directive
    FINRA = "finra"  # Financial Industry Regulatory Authority
    SEC = "sec"  # Securities and Exchange Commission
    GDPR = "gdpr"  # General Data Protection Regulation
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    SOX = "sox"  # Sarbanes-Oxley Act
    BASEL_III = "basel_iii"  # Basel III Regulatory Framework

# Audit Event Types
class AuditEventType(str, Enum):
    # Authentication Events
    USER_LOGIN = "auth.login"
    USER_LOGOUT = "auth.logout"
    LOGIN_FAILED = "auth.login_failed"
    PASSWORD_CHANGED = "auth.password_changed"
    MFA_ENABLED = "auth.mfa_enabled"
    
    # Trading Events
    ORDER_PLACED = "trading.order_placed"
    ORDER_MODIFIED = "trading.order_modified"
    ORDER_CANCELLED = "trading.order_cancelled"
    ORDER_EXECUTED = "trading.order_executed"
    
    # Data Access Events
    DATA_ACCESSED = "data.accessed"
    DATA_MODIFIED = "data.modified"
    DATA_DELETED = "data.deleted"
    DATA_EXPORTED = "data.exported"
    
    # Administrative Events
    USER_CREATED = "admin.user_created"
    USER_MODIFIED = "admin.user_modified"
    USER_DELETED = "admin.user_deleted"
    PERMISSION_CHANGED = "admin.permission_changed"
    
    # Compliance Events
    COMPLIANCE_CHECK = "compliance.check"
    COMPLIANCE_VIOLATION = "compliance.violation"
    SUSPICIOUS_ACTIVITY = "compliance.suspicious_activity"

@dataclass
class AuditRecord:
    """Immutable audit record"""
    record_id: str
    timestamp: datetime
    event_type: AuditEventType
    user_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    action: str
    outcome: str  # success, failure, error
    details: Dict[str, Any]
    risk_score: Optional[float] = None
    compliance_flags: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    def calculate_hash(self) -> str:
        """Calculate cryptographic hash of record"""
        record_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(record_str.encode()).hexdigest()

class EncryptionManager:
    """Manages encryption for sensitive audit data"""
    
    def __init__(self, master_key: str):
        # Derive encryption key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'octopus-audit-salt',  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self.cipher = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()

class AuditLogger:
    """Core audit logging service"""
    
    def __init__(self, 
                 db_pool: asyncpg.Pool,
                 encryption_manager: EncryptionManager):
        self.db_pool = db_pool
        self.encryption = encryption_manager
        self.buffer: List[AuditRecord] = []
        self.buffer_size = 100
        self.flush_interval = 5  # seconds
        self._running = False
        
    async def start(self):
        """Start the audit logger"""
        self._running = True
        # Start background flush task
        asyncio.create_task(self._flush_periodically())
        logger.info("Audit logger started")
    
    async def stop(self):
        """Stop the audit logger"""
        self._running = False
        # Flush remaining records
        await self._flush_buffer()
        logger.info("Audit logger stopped")
    
    async def log_event(self,
                       event_type: AuditEventType,
                       user_id: Optional[str],
                       action: str,
                       outcome: str,
                       resource_type: Optional[str] = None,
                       resource_id: Optional[str] = None,
                       details: Optional[Dict[str, Any]] = None,
                       ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None) -> str:
        """Log an audit event"""
        
        # Generate unique record ID
        record_id = f"audit_{datetime.utcnow().timestamp()}_{hashlib.md5(f'{event_type}{user_id}{action}'.encode()).hexdigest()[:8]}"
        
        # Create audit record
        record = AuditRecord(
            record_id=record_id,
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            details=details or {},
            risk_score=self._calculate_risk_score(event_type, outcome, details),
            compliance_flags=self._check_compliance_flags(event_type, details)
        )
        
        # Add to buffer
        self.buffer.append(record)
        
        # Flush if buffer is full
        if len(self.buffer) >= self.buffer_size:
            await self._flush_buffer()
        
        return record_id
    
    async def _flush_buffer(self):
        """Flush audit records to database"""
        if not self.buffer:
            return
            
        records_to_flush = self.buffer.copy()
        self.buffer.clear()
        
        try:
            async with self.db_pool.acquire() as conn:
                # Prepare batch insert
                values = []
                for record in records_to_flush:
                    # Encrypt sensitive details
                    encrypted_details = self.encryption.encrypt(
                        json.dumps(record.details)
                    )
                    
                    values.append((
                        record.record_id,
                        record.timestamp,
                        record.event_type.value,
                        record.user_id,
                        record.ip_address,
                        record.user_agent,
                        record.resource_type,
                        record.resource_id,
                        record.action,
                        record.outcome,
                        encrypted_details,
                        record.risk_score,
                        record.compliance_flags,
                        record.calculate_hash()
                    ))
                
                # Batch insert
                await conn.executemany("""
                    INSERT INTO audit_log (
                        record_id, timestamp, event_type, user_id,
                        ip_address, user_agent, resource_type, resource_id,
                        action, outcome, encrypted_details, risk_score,
                        compliance_flags, record_hash
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """, values)
                
                logger.info(f"Flushed {len(records_to_flush)} audit records")
                
        except Exception as e:
            logger.error(f"Failed to flush audit records: {e}")
            # Re-add to buffer for retry
            self.buffer.extend(records_to_flush)
    
    async def _flush_periodically(self):
        """Periodically flush buffer"""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            await self._flush_buffer()
    
    def _calculate_risk_score(self, 
                            event_type: AuditEventType,
                            outcome: str,
                            details: Optional[Dict[str, Any]]) -> float:
        """Calculate risk score for audit event"""
        score = 0.0
        
        # Failed authentication attempts
        if event_type == AuditEventType.LOGIN_FAILED:
            score += 0.3
            
        # Suspicious activities
        if event_type == AuditEventType.SUSPICIOUS_ACTIVITY:
            score += 0.8
            
        # Large transactions
        if details and 'amount' in details:
            amount = float(details['amount'])
            if amount > 100000:
                score += 0.5
            elif amount > 1000000:
                score += 0.8
                
        # After hours activity
        hour = datetime.now().hour
        if hour < 6 or hour > 22:
            score += 0.2
            
        return min(score, 1.0)
    
    def _check_compliance_flags(self,
                              event_type: AuditEventType,
                              details: Optional[Dict[str, Any]]) -> List[str]:
        """Check for compliance-related flags"""
        flags = []
        
        # GDPR - Data access/export
        if event_type in [AuditEventType.DATA_ACCESSED, AuditEventType.DATA_EXPORTED]:
            flags.append("GDPR_DATA_ACCESS")
            
        # FINRA - Large trades
        if event_type == AuditEventType.ORDER_EXECUTED and details:
            if details.get('amount', 0) > 50000:
                flags.append("FINRA_LARGE_TRADE")
                
        # SOX - Administrative changes
        if event_type in [AuditEventType.USER_MODIFIED, AuditEventType.PERMISSION_CHANGED]:
            flags.append("SOX_ADMIN_CHANGE")
            
        return flags

class ComplianceEngine:
    """Regulatory compliance checking engine"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.rules: Dict[ComplianceStandard, List[ComplianceRule]] = {}
        self._initialize_rules()
        
    def _initialize_rules(self):
        """Initialize compliance rules"""
        # MIFID II Rules
        self.rules[ComplianceStandard.MIFID_II] = [
            BestExecutionRule(),
            TransactionReportingRule(),
            ClientOrderHandlingRule()
        ]
        
        # FINRA Rules
        self.rules[ComplianceStandard.FINRA] = [
            PatternDayTradingRule(),
            MarginRequirementRule(),
            SuitabilityRule()
        ]
        
        # SEC Rules
        self.rules[ComplianceStandard.SEC] = [
            InsiderTradingRule(),
            MarketManipulationRule(),
            ShortSellingRule()
        ]
        
        # GDPR Rules
        self.rules[ComplianceStandard.GDPR] = [
            DataRetentionRule(),
            RightToErasureRule(),
            DataPortabilityRule()
        ]
    
    async def check_compliance(self,
                             standard: ComplianceStandard,
                             context: Dict[str, Any]) -> ComplianceResult:
        """Check compliance for specific standard"""
        
        violations = []
        warnings = []
        
        # Get rules for standard
        rules = self.rules.get(standard, [])
        
        # Check each rule
        for rule in rules:
            result = await rule.check(context)
            
            if result.status == ComplianceStatus.VIOLATION:
                violations.append(result)
            elif result.status == ComplianceStatus.WARNING:
                warnings.append(result)
        
        # Log compliance check
        await self.audit_logger.log_event(
            event_type=AuditEventType.COMPLIANCE_CHECK,
            user_id=context.get('user_id'),
            action=f"compliance_check_{standard.value}",
            outcome="completed",
            details={
                'standard': standard.value,
                'violations': len(violations),
                'warnings': len(warnings)
            }
        )
        
        # Return aggregated result
        if violations:
            return ComplianceResult(
                status=ComplianceStatus.VIOLATION,
                standard=standard,
                violations=violations,
                warnings=warnings
            )
        elif warnings:
            return ComplianceResult(
                status=ComplianceStatus.WARNING,
                standard=standard,
                violations=[],
                warnings=warnings
            )
        else:
            return ComplianceResult(
                status=ComplianceStatus.COMPLIANT,
                standard=standard,
                violations=[],
                warnings=[]
            )

# Compliance Rules Base Classes
class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"

@dataclass
class RuleResult:
    """Result of a compliance rule check"""
    rule_name: str
    status: ComplianceStatus
    message: str
    details: Optional[Dict[str, Any]] = None

@dataclass
class ComplianceResult:
    """Aggregated compliance result"""
    status: ComplianceStatus
    standard: ComplianceStandard
    violations: List[RuleResult]
    warnings: List[RuleResult]

class ComplianceRule:
    """Base class for compliance rules"""
    
    async def check(self, context: Dict[str, Any]) -> RuleResult:
        """Check if rule is satisfied"""
        raise NotImplementedError

# Example Compliance Rules
class BestExecutionRule(ComplianceRule):
    """MIFID II Best Execution Rule"""
    
    async def check(self, context: Dict[str, Any]) -> RuleResult:
        order = context.get('order')
        if not order:
            return RuleResult(
                rule_name="BestExecution",
                status=ComplianceStatus.COMPLIANT,
                message="No order to check"
            )
        
        # Check if best execution was achieved
        execution_price = order.get('execution_price', 0)
        best_available_price = context.get('best_available_price', 0)
        
        if execution_price > best_available_price * 1.001:  # 0.1% tolerance
            return RuleResult(
                rule_name="BestExecution",
                status=ComplianceStatus.VIOLATION,
                message="Order not executed at best available price",
                details={
                    'execution_price': execution_price,
                    'best_available_price': best_available_price,
                    'difference': execution_price - best_available_price
                }
            )
        
        return RuleResult(
            rule_name="BestExecution",
            status=ComplianceStatus.COMPLIANT,
            message="Best execution achieved"
        )

class PatternDayTradingRule(ComplianceRule):
    """FINRA Pattern Day Trading Rule"""
    
    async def check(self, context: Dict[str, Any]) -> RuleResult:
        account = context.get('account')
        if not account:
            return RuleResult(
                rule_name="PatternDayTrading",
                status=ComplianceStatus.COMPLIANT,
                message="No account to check"
            )
        
        # Check PDT requirements
        day_trades = account.get('day_trades_count', 0)
        account_value = account.get('value', 0)
        
        if day_trades >= 4 and account_value < 25000:
            return RuleResult(
                rule_name="PatternDayTrading",
                status=ComplianceStatus.VIOLATION,
                message="Pattern day trading violation - insufficient account value",
                details={
                    'day_trades': day_trades,
                    'account_value': account_value,
                    'required_value': 25000
                }
            )
        
        return RuleResult(
            rule_name="PatternDayTrading",
            status=ComplianceStatus.COMPLIANT,
            message="PDT requirements satisfied"
        )

class DataRetentionRule(ComplianceRule):
    """GDPR Data Retention Rule"""
    
    async def check(self, context: Dict[str, Any]) -> RuleResult:
        data_age_days = context.get('data_age_days', 0)
        data_type = context.get('data_type', 'general')
        
        # Different retention periods for different data types
        retention_limits = {
            'trading_data': 7 * 365,  # 7 years for financial records
            'personal_data': 3 * 365,  # 3 years for personal data
            'marketing_data': 365,  # 1 year for marketing
            'general': 5 * 365  # 5 years default
        }
        
        limit = retention_limits.get(data_type, retention_limits['general'])
        
        if data_age_days > limit:
            return RuleResult(
                rule_name="DataRetention",
                status=ComplianceStatus.WARNING,
                message=f"Data exceeds retention period for {data_type}",
                details={
                    'data_age_days': data_age_days,
                    'retention_limit_days': limit,
                    'data_type': data_type
                }
            )
        
        return RuleResult(
            rule_name="DataRetention",
            status=ComplianceStatus.COMPLIANT,
            message="Data within retention period"
        )

# Audit Query Service
class AuditQueryService:
    """Service for querying audit logs"""
    
    def __init__(self, 
                 db_pool: asyncpg.Pool,
                 encryption_manager: EncryptionManager):
        self.db_pool = db_pool
        self.encryption = encryption_manager
    
    async def query_by_user(self,
                           user_id: str,
                           start_date: datetime,
                           end_date: datetime,
                           event_types: Optional[List[AuditEventType]] = None) -> List[AuditRecord]:
        """Query audit logs by user"""
        
        query = """
            SELECT * FROM audit_log
            WHERE user_id = $1
            AND timestamp >= $2
            AND timestamp <= $3
        """
        
        params = [user_id, start_date, end_date]
        
        if event_types:
            query += " AND event_type = ANY($4)"
            params.append([et.value for et in event_types])
            
        query += " ORDER BY timestamp DESC"
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
        return [self._row_to_record(row) for row in rows]
    
    async def query_by_resource(self,
                              resource_type: str,
                              resource_id: str,
                              limit: int = 100) -> List[AuditRecord]:
        """Query audit logs by resource"""
        
        query = """
            SELECT * FROM audit_log
            WHERE resource_type = $1
            AND resource_id = $2
            ORDER BY timestamp DESC
            LIMIT $3
        """
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, resource_type, resource_id, limit)
            
        return [self._row_to_record(row) for row in rows]
    
    async def query_suspicious_activities(self,
                                        start_date: datetime,
                                        risk_threshold: float = 0.7) -> List[AuditRecord]:
        """Query suspicious activities based on risk score"""
        
        query = """
            SELECT * FROM audit_log
            WHERE risk_score >= $1
            AND timestamp >= $2
            ORDER BY risk_score DESC, timestamp DESC
        """
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, risk_threshold, start_date)
            
        return [self._row_to_record(row) for row in rows]
    
    def _row_to_record(self, row) -> AuditRecord:
        """Convert database row to AuditRecord"""
        
        # Decrypt details
        decrypted_details = json.loads(
            self.encryption.decrypt(row['encrypted_details'])
        )
        
        return AuditRecord(
            record_id=row['record_id'],
            timestamp=row['timestamp'],
            event_type=AuditEventType(row['event_type']),
            user_id=row['user_id'],
            ip_address=row['ip_address'],
            user_agent=row['user_agent'],
            resource_type=row['resource_type'],
            resource_id=row['resource_id'],
            action=row['action'],
            outcome=row['outcome'],
            details=decrypted_details,
            risk_score=row['risk_score'],
            compliance_flags=row['compliance_flags']
        )

# Initialize audit system
async def initialize_audit_system(db_pool: asyncpg.Pool, 
                                master_key: str) -> tuple[AuditLogger, ComplianceEngine, AuditQueryService]:
    """Initialize the audit and compliance system"""
    
    # Create encryption manager
    encryption = EncryptionManager(master_key)
    
    # Create audit logger
    audit_logger = AuditLogger(db_pool, encryption)
    await audit_logger.start()
    
    # Create compliance engine
    compliance_engine = ComplianceEngine(audit_logger)
    
    # Create query service
    query_service = AuditQueryService(db_pool, encryption)
    
    logger.info("Audit and compliance system initialized")
    
    return audit_logger, compliance_engine, query_service 