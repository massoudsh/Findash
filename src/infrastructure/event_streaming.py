"""
Octopus Trading Platform™ - Event Streaming Infrastructure
Enterprise-grade event streaming with Kafka and event sourcing
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Callable, Type
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from abc import ABC, abstractmethod
import aiokafka
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer
import avro.schema
from pydantic import BaseModel
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Event Types
class EventType(str, Enum):
    # Trading Events
    ORDER_CREATED = "order.created"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REJECTED = "order.rejected"
    
    # Market Data Events
    PRICE_UPDATE = "market.price_update"
    QUOTE_UPDATE = "market.quote_update"
    TRADE_EXECUTED = "market.trade_executed"
    
    # Portfolio Events
    POSITION_OPENED = "portfolio.position_opened"
    POSITION_CLOSED = "portfolio.position_closed"
    PORTFOLIO_REBALANCED = "portfolio.rebalanced"
    
    # Risk Events
    RISK_ALERT = "risk.alert"
    MARGIN_CALL = "risk.margin_call"
    STOP_LOSS_TRIGGERED = "risk.stop_loss_triggered"
    
    # System Events
    SERVICE_STARTED = "system.service_started"
    SERVICE_STOPPED = "system.service_stopped"
    ERROR_OCCURRED = "system.error_occurred"
    
    # User Events
    USER_LOGGED_IN = "user.logged_in"
    USER_LOGGED_OUT = "user.logged_out"
    USER_PROFILE_UPDATED = "user.profile_updated"

@dataclass
class EventMetadata:
    """Event metadata for tracking and auditing"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    source_service: str
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class Event(BaseModel):
    """Base event class"""
    metadata: EventMetadata
    payload: Dict[str, Any]
    
    class Config:
        arbitrary_types_allowed = True

class EventStore:
    """Event store for event sourcing"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.event_ttl = 86400 * 30  # 30 days
        
    async def append_event(self, aggregate_id: str, event: Event) -> bool:
        """Append event to aggregate's event stream"""
        try:
            # Store event in Redis
            event_key = f"events:{aggregate_id}:{event.metadata.event_id}"
            event_data = json.dumps({
                "metadata": event.metadata.to_dict(),
                "payload": event.payload
            })
            
            # Store event
            await self.redis.setex(event_key, self.event_ttl, event_data)
            
            # Add to aggregate's event list
            await self.redis.rpush(f"aggregate:{aggregate_id}", event_key)
            
            # Update aggregate version
            await self.redis.hincrby(f"aggregate:version:{aggregate_id}", "version", 1)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to append event: {e}")
            return False
    
    async def get_events(self, aggregate_id: str, from_version: int = 0) -> List[Event]:
        """Get events for aggregate from specific version"""
        try:
            # Get event keys
            event_keys = await self.redis.lrange(f"aggregate:{aggregate_id}", from_version, -1)
            
            events = []
            for key in event_keys:
                event_data = await self.redis.get(key)
                if event_data:
                    data = json.loads(event_data)
                    # Reconstruct event
                    metadata = EventMetadata(**data['metadata'])
                    metadata.timestamp = datetime.fromisoformat(data['metadata']['timestamp'])
                    event = Event(metadata=metadata, payload=data['payload'])
                    events.append(event)
                    
            return events
            
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return []
    
    async def get_aggregate_version(self, aggregate_id: str) -> int:
        """Get current version of aggregate"""
        version = await self.redis.hget(f"aggregate:version:{aggregate_id}", "version")
        return int(version) if version else 0

class EventProducer:
    """Kafka event producer with schema registry"""
    
    def __init__(self, 
                 bootstrap_servers: str = "localhost:9092",
                 schema_registry_url: str = "http://localhost:8081"):
        self.bootstrap_servers = bootstrap_servers
        self.schema_registry_url = schema_registry_url
        self.producer: Optional[AIOKafkaProducer] = None
        self.schema_registry: Optional[SchemaRegistryClient] = None
        self.serializers: Dict[str, AvroSerializer] = {}
        
    async def start(self):
        """Start the producer"""
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode(),
            compression_type="snappy",
            acks='all',  # Wait for all replicas
            retries=5,
            max_in_flight_requests_per_connection=5
        )
        await self.producer.start()
        
        # Initialize schema registry
        self.schema_registry = SchemaRegistryClient({
            'url': self.schema_registry_url
        })
        
        logger.info("Event producer started")
    
    async def stop(self):
        """Stop the producer"""
        if self.producer:
            await self.producer.stop()
            logger.info("Event producer stopped")
    
    async def send_event(self, 
                        topic: str,
                        event: Event,
                        key: Optional[str] = None,
                        partition: Optional[int] = None) -> bool:
        """Send event to Kafka topic"""
        try:
            # Serialize event
            event_data = {
                "metadata": event.metadata.to_dict(),
                "payload": event.payload
            }
            
            # Send to Kafka
            await self.producer.send_and_wait(
                topic,
                value=event_data,
                key=key.encode() if key else None,
                partition=partition,
                headers=[
                    ("event_type", event.metadata.event_type.value.encode()),
                    ("event_id", event.metadata.event_id.encode()),
                    ("source_service", event.metadata.source_service.encode())
                ]
            )
            
            logger.debug(f"Event sent: {event.metadata.event_id} to topic {topic}")
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to send event: {e}")
            return False
    
    async def send_batch(self, 
                        topic: str,
                        events: List[Event]) -> int:
        """Send batch of events"""
        sent_count = 0
        
        # Create batch
        batch = self.producer.create_batch()
        
        for event in events:
            event_data = {
                "metadata": event.metadata.to_dict(),
                "payload": event.payload
            }
            
            # Try to append to batch
            metadata = batch.append(
                key=None,
                value=json.dumps(event_data).encode(),
                timestamp=int(event.metadata.timestamp.timestamp() * 1000)
            )
            
            # If batch is full, send it
            if metadata is None:
                await self.producer.send_batch(batch, topic)
                sent_count += len(batch)
                
                # Create new batch
                batch = self.producer.create_batch()
                batch.append(
                    key=None,
                    value=json.dumps(event_data).encode(),
                    timestamp=int(event.metadata.timestamp.timestamp() * 1000)
                )
        
        # Send remaining batch
        if len(batch) > 0:
            await self.producer.send_batch(batch, topic)
            sent_count += len(batch)
            
        return sent_count

class EventConsumer:
    """Kafka event consumer with automatic offset management"""
    
    def __init__(self,
                 topics: List[str],
                 group_id: str,
                 bootstrap_servers: str = "localhost:9092",
                 auto_offset_reset: str = "latest"):
        self.topics = topics
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers
        self.auto_offset_reset = auto_offset_reset
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.handlers: Dict[EventType, List[Callable]] = {}
        self.running = False
        
    async def start(self):
        """Start the consumer"""
        self.consumer = AIOKafkaConsumer(
            *self.topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset=self.auto_offset_reset,
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            value_deserializer=lambda v: json.loads(v.decode())
        )
        await self.consumer.start()
        self.running = True
        logger.info(f"Event consumer started for topics: {self.topics}")
    
    async def stop(self):
        """Stop the consumer"""
        self.running = False
        if self.consumer:
            await self.consumer.stop()
            logger.info("Event consumer stopped")
    
    def register_handler(self, event_type: EventType, handler: Callable):
        """Register event handler"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type}")
    
    async def consume_events(self):
        """Main consumption loop"""
        while self.running:
            try:
                # Get messages in batches
                messages = await self.consumer.getmany(timeout_ms=1000, max_records=100)
                
                for topic_partition, records in messages.items():
                    for record in records:
                        await self._process_message(record)
                        
            except Exception as e:
                logger.error(f"Error consuming events: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, message):
        """Process individual message"""
        try:
            # Extract event data
            event_data = message.value
            
            # Reconstruct event
            metadata = EventMetadata(**event_data['metadata'])
            metadata.timestamp = datetime.fromisoformat(event_data['metadata']['timestamp'])
            event = Event(metadata=metadata, payload=event_data['payload'])
            
            # Call registered handlers
            if metadata.event_type in self.handlers:
                for handler in self.handlers[metadata.event_type]:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Handler error for {metadata.event_type}: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to process message: {e}")

class EventBus:
    """Central event bus for the application"""
    
    def __init__(self, 
                 producer: EventProducer,
                 event_store: EventStore):
        self.producer = producer
        self.event_store = event_store
        self.local_handlers: Dict[EventType, List[Callable]] = {}
        
    async def publish(self, 
                     event_type: EventType,
                     payload: Dict[str, Any],
                     source_service: str,
                     aggregate_id: Optional[str] = None,
                     correlation_id: Optional[str] = None,
                     user_id: Optional[str] = None) -> str:
        """Publish event to bus"""
        
        # Create event metadata
        event_id = str(uuid.uuid4())
        metadata = EventMetadata(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            source_service=source_service,
            correlation_id=correlation_id,
            user_id=user_id
        )
        
        # Create event
        event = Event(metadata=metadata, payload=payload)
        
        # Store in event store if aggregate_id provided
        if aggregate_id:
            await self.event_store.append_event(aggregate_id, event)
        
        # Determine topic based on event type
        topic = self._get_topic_for_event(event_type)
        
        # Send to Kafka
        success = await self.producer.send_event(topic, event, key=aggregate_id)
        
        # Call local handlers
        await self._call_local_handlers(event)
        
        return event_id if success else None
    
    def subscribe_local(self, event_type: EventType, handler: Callable):
        """Subscribe to events locally (in-process)"""
        if event_type not in self.local_handlers:
            self.local_handlers[event_type] = []
        self.local_handlers[event_type].append(handler)
    
    async def _call_local_handlers(self, event: Event):
        """Call local event handlers"""
        if event.metadata.event_type in self.local_handlers:
            for handler in self.local_handlers[event.metadata.event_type]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Local handler error: {e}")
    
    def _get_topic_for_event(self, event_type: EventType) -> str:
        """Determine Kafka topic for event type"""
        # Extract domain from event type
        domain = event_type.value.split('.')[0]
        return f"octopus.{domain}.events"

# Event Handlers Base Classes
class EventHandler(ABC):
    """Base class for event handlers"""
    
    @abstractmethod
    async def handle(self, event: Event):
        """Handle the event"""
        pass

class AggregateRoot(ABC):
    """Base class for aggregate roots in event sourcing"""
    
    def __init__(self, aggregate_id: str):
        self.aggregate_id = aggregate_id
        self.version = 0
        self.uncommitted_events: List[Event] = []
        
    def apply_event(self, event: Event):
        """Apply event to aggregate"""
        # Call appropriate handler method
        handler_name = f"_handle_{event.metadata.event_type.value.replace('.', '_')}"
        handler = getattr(self, handler_name, None)
        if handler:
            handler(event)
        self.version += 1
    
    def add_event(self, event_type: EventType, payload: Dict[str, Any], source_service: str):
        """Add new event to uncommitted events"""
        metadata = EventMetadata(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            source_service=source_service
        )
        event = Event(metadata=metadata, payload=payload)
        self.uncommitted_events.append(event)
        self.apply_event(event)
    
    def get_uncommitted_events(self) -> List[Event]:
        """Get uncommitted events"""
        return self.uncommitted_events
    
    def mark_events_committed(self):
        """Mark events as committed"""
        self.uncommitted_events.clear()

# Saga Pattern Implementation
class SagaStep:
    """Represents a step in a saga"""
    
    def __init__(self, 
                 name: str,
                 execute: Callable,
                 compensate: Callable):
        self.name = name
        self.execute = execute
        self.compensate = compensate

class Saga:
    """Saga orchestrator for distributed transactions"""
    
    def __init__(self, saga_id: str, event_bus: EventBus):
        self.saga_id = saga_id
        self.event_bus = event_bus
        self.steps: List[SagaStep] = []
        self.completed_steps: List[str] = []
        
    def add_step(self, step: SagaStep):
        """Add step to saga"""
        self.steps.append(step)
    
    async def execute(self) -> bool:
        """Execute saga"""
        try:
            # Execute all steps
            for step in self.steps:
                logger.info(f"Executing saga step: {step.name}")
                await step.execute()
                self.completed_steps.append(step.name)
                
                # Publish step completed event
                await self.event_bus.publish(
                    EventType.SYSTEM_EVENT,
                    {
                        "saga_id": self.saga_id,
                        "step": step.name,
                        "status": "completed"
                    },
                    "saga_orchestrator"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Saga failed at step {step.name}: {e}")
            # Compensate completed steps in reverse order
            await self.compensate()
            return False
    
    async def compensate(self):
        """Compensate completed steps"""
        for step_name in reversed(self.completed_steps):
            step = next((s for s in self.steps if s.name == step_name), None)
            if step:
                try:
                    logger.info(f"Compensating saga step: {step.name}")
                    await step.compensate()
                    
                    # Publish compensation event
                    await self.event_bus.publish(
                        EventType.SYSTEM_EVENT,
                        {
                            "saga_id": self.saga_id,
                            "step": step.name,
                            "status": "compensated"
                        },
                        "saga_orchestrator"
                    )
                except Exception as e:
                    logger.error(f"Compensation failed for {step.name}: {e}")

# Initialize components
async def initialize_event_streaming(redis_url: str = "redis://localhost:6379",
                                   kafka_servers: str = "localhost:9092") -> EventBus:
    """Initialize event streaming infrastructure"""
    
    # Initialize Redis for event store
    redis_client = await redis.from_url(redis_url)
    event_store = EventStore(redis_client)
    
    # Initialize Kafka producer
    producer = EventProducer(bootstrap_servers=kafka_servers)
    await producer.start()
    
    # Create event bus
    event_bus = EventBus(producer, event_store)
    
    logger.info("Event streaming infrastructure initialized")
    return event_bus 