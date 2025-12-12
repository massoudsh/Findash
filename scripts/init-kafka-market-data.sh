#!/bin/bash
# Initialize Kafka topic for market data streaming

set -e

KAFKA_BOOTSTRAP_SERVERS=${KAFKA_BOOTSTRAP_SERVERS:-"localhost:9092"}
TOPIC_NAME="market-data-stream"
PARTITIONS=${PARTITIONS:-3}
REPLICATION_FACTOR=${REPLICATION_FACTOR:-1}

echo "🔧 Initializing Kafka topic: $TOPIC_NAME"

# Wait for Kafka to be ready
echo "⏳ Waiting for Kafka to be ready..."
for i in {1..30}; do
    if docker exec octopus-kafka kafka-broker-api-versions --bootstrap-server localhost:9092 > /dev/null 2>&1; then
        echo "✅ Kafka is ready"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

# Confluent Kafka uses kafka-topics command (not .sh)
KAFKA_CMD="kafka-topics"

# Create topic if it doesn't exist
echo "📝 Creating topic: $TOPIC_NAME"
docker exec octopus-kafka $KAFKA_CMD \
    --bootstrap-server localhost:9092 \
    --create \
    --topic "$TOPIC_NAME" \
    --partitions "$PARTITIONS" \
    --replication-factor "$REPLICATION_FACTOR" \
    --if-not-exists 2>&1 || echo "Topic may already exist (or using auto-create)"

# Verify topic creation
echo "✅ Verifying topic..."
docker exec octopus-kafka $KAFKA_CMD \
    --bootstrap-server localhost:9092 \
    --describe \
    --topic "$TOPIC_NAME" 2>&1 || echo "Note: Topic will be auto-created on first message if KAFKA_AUTO_CREATE_TOPICS_ENABLE=true"

echo "🎉 Kafka topic '$TOPIC_NAME' is ready!"

