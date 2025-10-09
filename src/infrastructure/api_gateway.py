"""
Octopus Trading Platform™ - API Gateway Layer
Enterprise-grade API management with Kong integration
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import httpx
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from circuitbreaker import circuit
import consul
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Metrics
api_requests = Counter('api_gateway_requests_total', 'Total API requests', ['service', 'method', 'status'])
api_latency = Histogram('api_gateway_request_duration_seconds', 'API request latency', ['service', 'method'])
active_connections = Gauge('api_gateway_active_connections', 'Active connections')

@dataclass
class ServiceConfig:
    """Service configuration for registry"""
    name: str
    host: str
    port: int
    health_check_url: str
    weight: int = 1
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 60
    burst_size: int = 10
    window_size: int = 60

class LoadBalancerStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    IP_HASH = "ip_hash"
    RANDOM = "random"

class ServiceRegistry:
    """Service discovery and registration using Consul"""
    
    def __init__(self, consul_host: str = "localhost", consul_port: int = 8500):
        self.consul = consul.Consul(host=consul_host, port=consul_port)
        self.services: Dict[str, List[ServiceConfig]] = {}
        self._health_check_interval = 30
        
    async def register_service(self, service: ServiceConfig) -> bool:
        """Register service with Consul"""
        try:
            # Register with Consul
            self.consul.agent.service.register(
                name=service.name,
                service_id=f"{service.name}-{service.host}-{service.port}",
                address=service.host,
                port=service.port,
                tags=[f"version:{service.version}"],
                check=consul.Check.http(
                    service.health_check_url,
                    interval=f"{self._health_check_interval}s",
                    timeout="5s"
                )
            )
            
            # Update local registry
            if service.name not in self.services:
                self.services[service.name] = []
            self.services[service.name].append(service)
            
            logger.info(f"Registered service: {service.name} at {service.host}:{service.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {service.name}: {e}")
            return False
    
    async def discover_service(self, service_name: str) -> List[ServiceConfig]:
        """Discover available service instances"""
        try:
            # Query Consul for service instances
            _, instances = self.consul.health.service(service_name, passing=True)
            
            services = []
            for instance in instances:
                service = instance['Service']
                config = ServiceConfig(
                    name=service['Service'],
                    host=service['Address'],
                    port=service['Port'],
                    health_check_url=f"http://{service['Address']}:{service['Port']}/health",
                    metadata=service.get('Meta', {})
                )
                services.append(config)
                
            return services
            
        except Exception as e:
            logger.error(f"Service discovery failed for {service_name}: {e}")
            return []
    
    async def deregister_service(self, service_id: str) -> bool:
        """Deregister service from Consul"""
        try:
            self.consul.agent.service.deregister(service_id)
            logger.info(f"Deregistered service: {service_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to deregister service {service_id}: {e}")
            return False

class LoadBalancer:
    """Load balancer with multiple strategies"""
    
    def __init__(self, strategy: LoadBalancerStrategy = LoadBalancerStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.current_index: Dict[str, int] = {}
        self.connection_count: Dict[str, int] = {}
        
    def select_instance(self, service_name: str, instances: List[ServiceConfig], 
                       client_ip: Optional[str] = None) -> Optional[ServiceConfig]:
        """Select service instance based on strategy"""
        if not instances:
            return None
            
        if self.strategy == LoadBalancerStrategy.ROUND_ROBIN:
            return self._round_robin(service_name, instances)
        elif self.strategy == LoadBalancerStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(service_name, instances)
        elif self.strategy == LoadBalancerStrategy.LEAST_CONNECTIONS:
            return self._least_connections(instances)
        elif self.strategy == LoadBalancerStrategy.IP_HASH:
            return self._ip_hash(instances, client_ip)
        else:  # RANDOM
            import random
            return random.choice(instances)
    
    def _round_robin(self, service_name: str, instances: List[ServiceConfig]) -> ServiceConfig:
        """Round-robin selection"""
        if service_name not in self.current_index:
            self.current_index[service_name] = 0
            
        index = self.current_index[service_name]
        instance = instances[index % len(instances)]
        self.current_index[service_name] = (index + 1) % len(instances)
        
        return instance
    
    def _weighted_round_robin(self, service_name: str, instances: List[ServiceConfig]) -> ServiceConfig:
        """Weighted round-robin based on instance weight"""
        weighted_instances = []
        for instance in instances:
            weighted_instances.extend([instance] * instance.weight)
        return self._round_robin(service_name, weighted_instances)
    
    def _least_connections(self, instances: List[ServiceConfig]) -> ServiceConfig:
        """Select instance with least active connections"""
        min_connections = float('inf')
        selected = instances[0]
        
        for instance in instances:
            key = f"{instance.host}:{instance.port}"
            connections = self.connection_count.get(key, 0)
            if connections < min_connections:
                min_connections = connections
                selected = instance
                
        return selected
    
    def _ip_hash(self, instances: List[ServiceConfig], client_ip: Optional[str]) -> ServiceConfig:
        """Consistent hashing based on client IP"""
        if not client_ip:
            return instances[0]
            
        hash_value = hash(client_ip)
        return instances[hash_value % len(instances)]

class CircuitBreakerManager:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.breakers: Dict[str, circuit] = {}
        
    def get_breaker(self, service_name: str) -> circuit:
        """Get or create circuit breaker for service"""
        if service_name not in self.breakers:
            self.breakers[service_name] = circuit(
                failure_threshold=self.failure_threshold,
                recovery_timeout=self.recovery_timeout,
                expected_exception=Exception
            )
        return self.breakers[service_name]

class APIGateway:
    """Main API Gateway implementation"""
    
    def __init__(self, 
                 service_registry: ServiceRegistry,
                 load_balancer: LoadBalancer,
                 circuit_breaker: CircuitBreakerManager):
        self.registry = service_registry
        self.load_balancer = load_balancer
        self.circuit_breaker = circuit_breaker
        self.client = httpx.AsyncClient(timeout=30.0)
        self.request_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
        
    async def route_request(self, 
                           service_name: str,
                           path: str,
                           method: str = "GET",
                           headers: Optional[Dict] = None,
                           body: Optional[Any] = None,
                           client_ip: Optional[str] = None) -> Response:
        """Route request to appropriate service"""
        
        # Record metrics
        start_time = time.time()
        active_connections.inc()
        
        try:
            # Discover available instances
            instances = await self.registry.discover_service(service_name)
            if not instances:
                raise HTTPException(status_code=503, detail=f"No healthy instances of {service_name}")
            
            # Select instance using load balancer
            instance = self.load_balancer.select_instance(service_name, instances, client_ip)
            if not instance:
                raise HTTPException(status_code=503, detail=f"Failed to select instance for {service_name}")
            
            # Build target URL
            target_url = f"http://{instance.host}:{instance.port}{path}"
            
            # Get circuit breaker for service
            breaker = self.circuit_breaker.get_breaker(service_name)
            
            # Make request with circuit breaker
            @breaker
            async def make_request():
                response = await self.client.request(
                    method=method,
                    url=target_url,
                    headers=headers,
                    json=body if body else None
                )
                return response
            
            # Execute request
            response = await make_request()
            
            # Record success metrics
            api_requests.labels(service=service_name, method=method, status=response.status_code).inc()
            api_latency.labels(service=service_name, method=method).observe(time.time() - start_time)
            
            # Return response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            
        except circuit.CircuitBreakerError:
            # Circuit is open
            api_requests.labels(service=service_name, method=method, status=503).inc()
            raise HTTPException(status_code=503, detail=f"Service {service_name} is temporarily unavailable")
            
        except Exception as e:
            # Record failure metrics
            api_requests.labels(service=service_name, method=method, status=500).inc()
            logger.error(f"Gateway error for {service_name}: {e}")
            raise HTTPException(status_code=500, detail="Internal gateway error")
            
        finally:
            active_connections.dec()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all registered services"""
        health_status = {
            "gateway": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {}
        }
        
        for service_name, instances in self.registry.services.items():
            service_health = {
                "total_instances": len(instances),
                "healthy_instances": 0,
                "instances": []
            }
            
            for instance in instances:
                try:
                    response = await self.client.get(instance.health_check_url, timeout=5.0)
                    is_healthy = response.status_code == 200
                    if is_healthy:
                        service_health["healthy_instances"] += 1
                        
                    service_health["instances"].append({
                        "host": instance.host,
                        "port": instance.port,
                        "healthy": is_healthy,
                        "version": instance.version
                    })
                except Exception as e:
                    service_health["instances"].append({
                        "host": instance.host,
                        "port": instance.port,
                        "healthy": False,
                        "error": str(e)
                    })
            
            health_status["services"][service_name] = service_health
            
        return health_status

class APIGatewayMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for API Gateway integration"""
    
    def __init__(self, app, gateway: APIGateway):
        super().__init__(app)
        self.gateway = gateway
        
    async def dispatch(self, request: Request, call_next):
        # Extract service name from path
        path_parts = request.url.path.strip("/").split("/")
        if len(path_parts) < 2:
            return await call_next(request)
            
        # Check if this is a gateway route
        if path_parts[0] != "gateway":
            return await call_next(request)
            
        service_name = path_parts[1]
        service_path = "/" + "/".join(path_parts[2:]) if len(path_parts) > 2 else "/"
        
        # Get request body
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.json()
            
        # Route through gateway
        return await self.gateway.route_request(
            service_name=service_name,
            path=service_path,
            method=request.method,
            headers=dict(request.headers),
            body=body,
            client_ip=request.client.host
        )

# Kong Integration
class KongAdapter:
    """Kong API Gateway adapter for advanced features"""
    
    def __init__(self, kong_admin_url: str = "http://localhost:8001"):
        self.admin_url = kong_admin_url
        self.client = httpx.AsyncClient()
        
    async def create_service(self, name: str, url: str) -> Dict[str, Any]:
        """Create service in Kong"""
        response = await self.client.post(
            f"{self.admin_url}/services",
            json={
                "name": name,
                "url": url,
                "retries": 5,
                "connect_timeout": 60000,
                "write_timeout": 60000,
                "read_timeout": 60000
            }
        )
        return response.json()
    
    async def create_route(self, service_name: str, paths: List[str]) -> Dict[str, Any]:
        """Create route for service"""
        response = await self.client.post(
            f"{self.admin_url}/services/{service_name}/routes",
            json={
                "paths": paths,
                "strip_path": True,
                "preserve_host": False
            }
        )
        return response.json()
    
    async def enable_plugin(self, service_name: str, plugin_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Enable plugin for service"""
        response = await self.client.post(
            f"{self.admin_url}/services/{service_name}/plugins",
            json={
                "name": plugin_name,
                "config": config
            }
        )
        return response.json()
    
    async def setup_rate_limiting(self, service_name: str, rate_limit: RateLimitConfig):
        """Setup rate limiting for service"""
        return await self.enable_plugin(
            service_name,
            "rate-limiting",
            {
                "minute": rate_limit.requests_per_minute,
                "policy": "local",
                "fault_tolerant": True
            }
        )
    
    async def setup_authentication(self, service_name: str):
        """Setup JWT authentication for service"""
        return await self.enable_plugin(
            service_name,
            "jwt",
            {
                "claims_to_verify": ["exp"],
                "key_claim_name": "iss",
                "secret_is_base64": False
            }
        )

# Initialize components
service_registry = ServiceRegistry()
load_balancer = LoadBalancer(LoadBalancerStrategy.WEIGHTED_ROUND_ROBIN)
circuit_breaker_manager = CircuitBreakerManager()
api_gateway = APIGateway(service_registry, load_balancer, circuit_breaker_manager)
kong_adapter = KongAdapter()

# Export for use in FastAPI
gateway_middleware = lambda app: APIGatewayMiddleware(app, api_gateway) 