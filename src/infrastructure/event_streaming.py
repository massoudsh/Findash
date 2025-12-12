"""
Octopus Trading Platform™ - Event Streaming Infrastructure

This module provides a Redis-first eventing layer:
- Redis Streams for durable event transport (XADD / XREADGROUP)
- Redis-backed event store utilities for event sourcing

Notes:
- Streams are keyed by "topic" (stream key)
- Consumers use Redis consumer groups for scalable fanout
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import redis.asyncio as redis
from pydantic import BaseModel

logger = logging.getLogger(__name__)


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
        data["timestamp"] = self.timestamp.isoformat()
        data["event_type"] = self.event_type.value
        return data


class Event(BaseModel):
    """Base event class"""

    metadata: EventMetadata
    payload: Dict[str, Any]

    class Config:
        arbitrary_types_allowed = True


class EventStore:
    """Event store for event sourcing (Redis-backed)"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.event_ttl = 86400 * 30  # 30 days

    async def append_event(self, aggregate_id: str, event: Event) -> bool:
        try:
            event_key = f"events:{aggregate_id}:{event.metadata.event_id}"
            event_data = json.dumps({"metadata": event.metadata.to_dict(), "payload": event.payload})

            await self.redis.setex(event_key, self.event_ttl, event_data)
            await self.redis.rpush(f"aggregate:{aggregate_id}", event_key)
            await self.redis.hincrby(f"aggregate:version:{aggregate_id}", "version", 1)
            return True
        except Exception as e:
            logger.error(f"Failed to append event: {e}")
            return False

    async def get_events(self, aggregate_id: str, from_version: int = 0) -> List[Event]:
        try:
            event_keys = await self.redis.lrange(f"aggregate:{aggregate_id}", from_version, -1)
            events: List[Event] = []

            for key in event_keys:
                event_data = await self.redis.get(key)
                if not event_data:
                    continue

                data = json.loads(event_data)
                md = data["metadata"]
                metadata = EventMetadata(
                    event_id=md["event_id"],
                    event_type=EventType(md["event_type"]),
                    timestamp=datetime.fromisoformat(md["timestamp"]),
                    source_service=md["source_service"],
                    correlation_id=md.get("correlation_id"),
                    user_id=md.get("user_id"),
                    version=md.get("version", "1.0"),
                )
                events.append(Event(metadata=metadata, payload=data["payload"]))

            return events
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return []

    async def get_aggregate_version(self, aggregate_id: str) -> int:
        version = await self.redis.hget(f"aggregate:version:{aggregate_id}", "version")
        return int(version) if version else 0


class EventTransport(ABC):
    """Abstract interface for event transport"""

    @abstractmethod
    async def publish(self, stream: str, event: Event, *, maxlen: int = 100000) -> str:
        raise NotImplementedError

    @abstractmethod
    async def consume_forever(
        self,
        stream: str,
        group: str,
        consumer: str,
        handler: Callable[[Event], "asyncio.Future"],
        *,
        block_ms: int = 1000,
        count: int = 100,
    ) -> None:
        raise NotImplementedError


class RedisStreamTransport(EventTransport):
    """Redis Streams transport (durable)"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def ensure_group(self, stream: str, group: str) -> None:
        try:
            await self.redis.xgroup_create(stream, group, id="0-0", mkstream=True)
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def publish(self, stream: str, event: Event, *, maxlen: int = 100000) -> str:
        payload = json.dumps({"metadata": event.metadata.to_dict(), "payload": event.payload})
        # Store the whole event as one field for simplicity
        msg_id = await self.redis.xadd(stream, {"event": payload}, maxlen=maxlen, approximate=True)
        return str(msg_id)

    async def consume_forever(
        self,
        stream: str,
        group: str,
        consumer: str,
        handler: Callable[[Event], "asyncio.Future"],
        *,
        block_ms: int = 1000,
        count: int = 100,
    ) -> None:
        import asyncio

        await self.ensure_group(stream, group)

        while True:
            resp = await self.redis.xreadgroup(
                groupname=group,
                consumername=consumer,
                streams={stream: ">"},
                count=count,
                block=block_ms,
            )

            if not resp:
                continue

            for _stream, messages in resp:
                for msg_id, fields in messages:
                    raw = fields.get("event")
                    if not raw:
                        await self.redis.xack(stream, group, msg_id)
                        continue

                    try:
                        data = json.loads(raw)
                        md = data["metadata"]
                        metadata = EventMetadata(
                            event_id=md["event_id"],
                            event_type=EventType(md["event_type"]),
                            timestamp=datetime.fromisoformat(md["timestamp"]),
                            source_service=md["source_service"],
                            correlation_id=md.get("correlation_id"),
                            user_id=md.get("user_id"),
                            version=md.get("version", "1.0"),
                        )
                        event = Event(metadata=metadata, payload=data["payload"])
                        await handler(event)
                        await self.redis.xack(stream, group, msg_id)
                    except Exception as e:
                        logger.error(f"Failed to handle event {msg_id}: {e}")
                        await asyncio.sleep(0.5)


def new_event(
    *,
    event_type: EventType,
    payload: Dict[str, Any],
    source_service: str,
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Event:
    metadata = EventMetadata(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        timestamp=datetime.utcnow(),
        source_service=source_service,
        correlation_id=correlation_id,
        user_id=user_id,
    )
    return Event(metadata=metadata, payload=payload)
