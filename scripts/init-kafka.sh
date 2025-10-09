#!/bin/bash

# 🚀 Kafka Topics Initialization Script - Octopus Trading Platform™
# This script sets up Kafka topics, partitions, and configurations for the trading platform

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
KAFKA_BOOTSTRAP_SERVERS="${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092}"
KAFKA_HOME="${KAFKA_HOME:-/opt/kafka}"
ZOOKEEPER_CONNECT="${ZOOKEEPER_CONNECT:-localhost:2181}"
WAIT_TIMEOUT=60
REPLICATION_FACTOR="${REPLICATION_FACTOR:-1}"
DEFAULT_PARTITIONS="${DEFAULT_PARTITIONS:-3}"

echo -e "${BLUE}🚀 Initializing Kafka Topics for Octopus Trading Platform${NC}"

# Function to wait for Kafka to be ready
wait_for_kafka() {
    echo -e "${YELLOW}⏳ Waiting for Kafka to be ready...${NC}"
    local count=0
    while [[ $count -lt $WAIT_TIMEOUT ]]; do
        if kafka-topics.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" --list > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Kafka is ready${NC}"
            return 0
        fi
        sleep 2
        ((count+=2))
    done
    echo -e "${RED}❌ Kafka is not ready after $WAIT_TIMEOUT seconds${NC}"
    exit 1
}

# Function to create a topic
create_topic() {
    local topic_name=$1
    local partitions=${2:-$DEFAULT_PARTITIONS}
    local replication=${3:-$REPLICATION_FACTOR}
    local config=${4:-""}
    
    echo -e "${BLUE}📊 Creating/updating topic: $topic_name${NC}"
    
    # Check if topic exists
    if kafka-topics.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" --describe --topic "$topic_name" > /dev/null 2>&1; then
        echo -e "${YELLOW}🔄 Topic $topic_name already exists, checking configuration...${NC}"
        
        # Update configuration if provided
        if [[ -n "$config" ]]; then
            echo -e "${BLUE}⚙️  Updating topic configuration for $topic_name${NC}"
            kafka-configs.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" \
                --entity-type topics --entity-name "$topic_name" \
                --alter --add-config "$config"
        fi
    else
        echo -e "${GREEN}🆕 Creating new topic: $topic_name${NC}"
        local create_cmd="kafka-topics.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS --create --topic $topic_name --partitions $partitions --replication-factor $replication"
        
        if [[ -n "$config" ]]; then
            create_cmd="$create_cmd --config $config"
        fi
        
        eval "$create_cmd"
    fi
    echo ""
}

# Function to create consumer group
create_consumer_group() {
    local group_name=$1
    local topic_name=$2
    
    echo -e "${BLUE}👥 Ensuring consumer group: $group_name${NC}"
    # Consumer groups are created automatically when first consumer connects
    # This is just for documentation purposes
    echo -e "  • Group: $group_name"
    echo -e "  • Topic: $topic_name"
    echo ""
}

# Function to set topic ACLs (if security is enabled)
set_topic_acls() {
    local topic_name=$1
    local user=${2:-"trading-service"}
    
    echo -e "${BLUE}🔐 Setting ACLs for topic: $topic_name (user: $user)${NC}"
    
    # Check if Kafka has security enabled (this will fail gracefully if not)
    if kafka-acls.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" --list > /dev/null 2>&1; then
        # Allow read/write access to the topic
        kafka-acls.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" \
            --add --allow-principal "User:$user" \
            --operation Read --operation Write --topic "$topic_name"
        
        # Allow group operations for consumers
        kafka-acls.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" \
            --add --allow-principal "User:$user" \
            --operation Read --group "*"
    else
        echo -e "${YELLOW}⚠️  Security not configured, skipping ACLs${NC}"
    fi
    echo ""
}

# Start initialization
wait_for_kafka

echo -e "${BLUE}🏗️  Creating Trading Platform Topics...${NC}"

# Real-time Market Data Topics
create_topic "market-data.quotes" 6 $REPLICATION_FACTOR "retention.ms=300000,compression.type=lz4,cleanup.policy=delete"
create_topic "market-data.trades" 6 $REPLICATION_FACTOR "retention.ms=300000,compression.type=lz4,cleanup.policy=delete"
create_topic "market-data.orderbook" 12 $REPLICATION_FACTOR "retention.ms=60000,compression.type=lz4,cleanup.policy=delete"
create_topic "market-data.candles" 3 $REPLICATION_FACTOR "retention.ms=604800000,compression.type=gzip,cleanup.policy=delete"

# Portfolio & Trading Topics  
create_topic "portfolios.events" 3 $REPLICATION_FACTOR "retention.ms=2592000000,compression.type=gzip,cleanup.policy=compact"
create_topic "portfolios.snapshots" 3 $REPLICATION_FACTOR "retention.ms=2592000000,compression.type=gzip,cleanup.policy=delete"
create_topic "orders.events" 6 $REPLICATION_FACTOR "retention.ms=2592000000,compression.type=gzip,cleanup.policy=compact"
create_topic "orders.executions" 6 $REPLICATION_FACTOR "retention.ms=2592000000,compression.type=gzip,cleanup.policy=delete"
create_topic "positions.updates" 3 $REPLICATION_FACTOR "retention.ms=2592000000,compression.type=gzip,cleanup.policy=compact"

# Risk Management Topics
create_topic "risk.calculations" 3 $REPLICATION_FACTOR "retention.ms=604800000,compression.type=gzip,cleanup.policy=delete"
create_topic "risk.alerts" 1 $REPLICATION_FACTOR "retention.ms=2592000000,compression.type=gzip,cleanup.policy=delete"
create_topic "risk.violations" 1 $REPLICATION_FACTOR "retention.ms=2592000000,compression.type=gzip,cleanup.policy=delete"

# ML/AI Topics
create_topic "ml.predictions" 3 $REPLICATION_FACTOR "retention.ms=604800000,compression.type=gzip,cleanup.policy=delete"
create_topic "ml.features" 6 $REPLICATION_FACTOR "retention.ms=86400000,compression.type=lz4,cleanup.policy=delete"
create_topic "ml.model-updates" 1 $REPLICATION_FACTOR "retention.ms=2592000000,compression.type=gzip,cleanup.policy=compact"
create_topic "ml.backtests" 2 $REPLICATION_FACTOR "retention.ms=2592000000,compression.type=gzip,cleanup.policy=delete"

# Alternative Data Topics
create_topic "alt-data.news" 3 $REPLICATION_FACTOR "retention.ms=604800000,compression.type=gzip,cleanup.policy=delete"
create_topic "alt-data.social" 3 $REPLICATION_FACTOR "retention.ms=86400000,compression.type=gzip,cleanup.policy=delete"
create_topic "alt-data.sentiment" 2 $REPLICATION_FACTOR "retention.ms=604800000,compression.type=gzip,cleanup.policy=delete"
create_topic "alt-data.economic" 1 $REPLICATION_FACTOR "retention.ms=2592000000,compression.type=gzip,cleanup.policy=delete"

# User & Notification Topics
create_topic "users.events" 2 $REPLICATION_FACTOR "retention.ms=2592000000,compression.type=gzip,cleanup.policy=compact"
create_topic "notifications.alerts" 2 $REPLICATION_FACTOR "retention.ms=604800000,compression.type=gzip,cleanup.policy=delete"
create_topic "notifications.emails" 1 $REPLICATION_FACTOR "retention.ms=86400000,compression.type=gzip,cleanup.policy=delete"
create_topic "notifications.push" 2 $REPLICATION_FACTOR "retention.ms=86400000,compression.type=gzip,cleanup.policy=delete"

# System & Infrastructure Topics
create_topic "system.metrics" 3 $REPLICATION_FACTOR "retention.ms=604800000,compression.type=lz4,cleanup.policy=delete"
create_topic "system.logs" 6 $REPLICATION_FACTOR "retention.ms=604800000,compression.type=gzip,cleanup.policy=delete"
create_topic "system.audit" 2 $REPLICATION_FACTOR "retention.ms=7776000000,compression.type=gzip,cleanup.policy=delete"
create_topic "system.health" 1 $REPLICATION_FACTOR "retention.ms=86400000,compression.type=lz4,cleanup.policy=delete"

# Dead Letter Queue Topics
create_topic "dlq.market-data" 1 $REPLICATION_FACTOR "retention.ms=2592000000,compression.type=gzip,cleanup.policy=delete"
create_topic "dlq.orders" 1 $REPLICATION_FACTOR "retention.ms=2592000000,compression.type=gzip,cleanup.policy=delete"
create_topic "dlq.notifications" 1 $REPLICATION_FACTOR "retention.ms=604800000,compression.type=gzip,cleanup.policy=delete"
create_topic "dlq.general" 1 $REPLICATION_FACTOR "retention.ms=2592000000,compression.type=gzip,cleanup.policy=delete"

# Schema Registry Topics (if using Confluent Schema Registry)
if [[ "${USE_SCHEMA_REGISTRY:-false}" == "true" ]]; then
    echo -e "${BLUE}📋 Creating Schema Registry topics...${NC}"
    create_topic "_schemas" 1 $REPLICATION_FACTOR "retention.ms=-1,cleanup.policy=compact,compression.type=uncompressed"
fi

echo -e "${BLUE}👥 Setting up Consumer Groups...${NC}"

# Real-time processors
create_consumer_group "market-data-processor" "market-data.quotes"
create_consumer_group "orderbook-processor" "market-data.orderbook"
create_consumer_group "trade-processor" "market-data.trades"

# Portfolio services
create_consumer_group "portfolio-service" "portfolios.events"
create_consumer_group "position-service" "positions.updates"
create_consumer_group "order-service" "orders.events"

# Risk management
create_consumer_group "risk-calculator" "risk.calculations"
create_consumer_group "risk-monitor" "risk.alerts"

# ML services
create_consumer_group "ml-feature-extractor" "ml.features"
create_consumer_group "ml-predictor" "ml.predictions"

# Notification services
create_consumer_group "notification-dispatcher" "notifications.alerts"
create_consumer_group "email-service" "notifications.emails"

# Monitoring & Analytics
create_consumer_group "metrics-collector" "system.metrics"
create_consumer_group "log-aggregator" "system.logs"
create_consumer_group "audit-processor" "system.audit"

echo -e "${BLUE}🔍 Verifying topic creation...${NC}"

# List all topics and their configurations
echo -e "${GREEN}📋 Created topics:${NC}"
kafka-topics.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" --list | sort

echo ""
echo -e "${BLUE}📊 Topic details:${NC}"

# Show detailed info for key topics
key_topics=("market-data.quotes" "orders.events" "portfolios.events" "risk.alerts")
for topic in "${key_topics[@]}"; do
    echo -e "${YELLOW}Topic: $topic${NC}"
    kafka-topics.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" --describe --topic "$topic"
    echo ""
done

echo -e "${BLUE}⚙️  Creating Kafka Connect connectors (if available)...${NC}"

# Function to create connector (if Kafka Connect is available)
create_connector() {
    local connector_name=$1
    local connector_config=$2
    
    if command -v kafka-connect-standalone.sh > /dev/null 2>&1; then
        echo -e "${GREEN}🔗 Creating connector: $connector_name${NC}"
        # This would typically be done via REST API in a real deployment
        echo "Connector config saved to: /tmp/octopus-$connector_name.json"
        echo "$connector_config" > "/tmp/octopus-$connector_name.json"
    else
        echo -e "${YELLOW}⚠️  Kafka Connect not available, skipping connectors${NC}"
    fi
}

# PostgreSQL source connector for audit logs
postgres_connector='{
  "name": "octopus-postgres-audit-source",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "db",
    "database.port": "5432",
    "database.user": "octopus_app",
    "database.password": "secure_password",
    "database.dbname": "trading_db",
    "database.server.name": "octopus-db",
    "table.include.list": "public.audit_log,public.security_events",
    "topic.prefix": "audit",
    "schema.history.internal.kafka.bootstrap.servers": "'"$KAFKA_BOOTSTRAP_SERVERS"'",
    "schema.history.internal.kafka.topic": "audit.schema-changes"
  }
}'

create_connector "postgres-audit-source" "$postgres_connector"

# Elasticsearch sink connector for search
elasticsearch_connector='{
  "name": "octopus-elasticsearch-sink",
  "config": {
    "connector.class": "io.confluent.connect.elasticsearch.ElasticsearchSinkConnector",
    "topics": "alt-data.news,system.logs",
    "connection.url": "http://elasticsearch:9200",
    "type.name": "_doc",
    "key.ignore": "true",
    "schema.ignore": "true"
  }
}'

create_connector "elasticsearch-sink" "$elasticsearch_connector"

echo -e "${BLUE}🎯 Setting up monitoring topics...${NC}"

# JMX metrics topic for Kafka monitoring
create_topic "kafka.metrics" 1 $REPLICATION_FACTOR "retention.ms=86400000,compression.type=lz4,cleanup.policy=delete"

# Lag monitoring topic
create_topic "kafka.consumer-lag" 1 $REPLICATION_FACTOR "retention.ms=86400000,compression.type=lz4,cleanup.policy=delete"

echo -e "${GREEN}✅ Kafka topics initialization completed successfully!${NC}"

echo -e "${BLUE}📋 Summary:${NC}"
echo -e "  • Total Topics Created: $(kafka-topics.sh --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" --list | wc -l)"
echo -e "  • Default Partitions: $DEFAULT_PARTITIONS"
echo -e "  • Replication Factor: $REPLICATION_FACTOR"
echo -e "  • Bootstrap Servers: $KAFKA_BOOTSTRAP_SERVERS"

echo -e "${BLUE}🏷️  Topic Categories:${NC}"
echo -e "  • Market Data: 4 topics (quotes, trades, orderbook, candles)"
echo -e "  • Portfolio & Trading: 5 topics (events, snapshots, orders, executions, positions)"
echo -e "  • Risk Management: 3 topics (calculations, alerts, violations)"
echo -e "  • ML/AI: 4 topics (predictions, features, models, backtests)"
echo -e "  • Alternative Data: 4 topics (news, social, sentiment, economic)"
echo -e "  • User & Notifications: 4 topics (users, alerts, emails, push)"
echo -e "  • System: 4 topics (metrics, logs, audit, health)"
echo -e "  • Dead Letter Queues: 4 topics (market-data, orders, notifications, general)"

echo -e "${BLUE}⚙️  Retention Policies:${NC}"
echo -e "  • Real-time data: 5 minutes - 1 hour"
echo -e "  • Market data: 7 days"
echo -e "  • Trading data: 30 days"
echo -e "  • Audit data: 90 days"
echo -e "  • System logs: 7 days"

echo -e "${BLUE}🔄 Key Consumer Groups:${NC}"
echo -e "  • market-data-processor: Real-time quote processing"
echo -e "  • portfolio-service: Portfolio management events"
echo -e "  • risk-calculator: Risk metric calculations"
echo -e "  • ml-predictor: ML prediction processing"
echo -e "  • notification-dispatcher: Alert notifications"

echo -e "${BLUE}📊 Monitoring Commands:${NC}"
echo -e "  • List topics: kafka-topics.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS --list"
echo -e "  • Consumer lag: kafka-consumer-groups.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS --describe --all-groups"
echo -e "  • Topic details: kafka-topics.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS --describe --topic <topic-name>"

echo -e "${BLUE}📝 Next steps:${NC}"
echo -e "  1. Configure your application producers and consumers"
echo -e "  2. Set up Kafka Connect connectors for data integration"
echo -e "  3. Configure monitoring with Kafka Manager or Confluent Control Center"
echo -e "  4. Set up alerting for consumer lag and partition health"
echo -e "  5. Test message flow with sample data"

echo -e "${BLUE}🔧 Useful Scripts:${NC}"
echo -e "  • Test producer: kafka-console-producer.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS --topic market-data.quotes"
echo -e "  • Test consumer: kafka-console-consumer.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVERS --topic market-data.quotes --from-beginning"

echo -e "${GREEN}🚀 Kafka initialization script completed!${NC}" 