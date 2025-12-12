#!/usr/bin/env python3
"""
Create Kafka topic for market data streaming
Alternative method if kafka-topics.sh is not available
"""

import sys
import os
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

def create_topic(bootstrap_servers: str, topic_name: str, partitions: int = 3, replication_factor: int = 1):
    """Create Kafka topic"""
    try:
        admin = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id='topic-creator'
        )
        
        topic = NewTopic(
            name=topic_name,
            num_partitions=partitions,
            replication_factor=replication_factor
        )
        
        admin.create_topics([topic])
        print(f"✅ Topic '{topic_name}' created successfully")
        return True
        
    except TopicAlreadyExistsError:
        print(f"ℹ️  Topic '{topic_name}' already exists")
        return True
    except Exception as e:
        print(f"❌ Error creating topic: {e}")
        return False
    finally:
        if 'admin' in locals():
            admin.close()

if __name__ == '__main__':
    bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    topic_name = os.getenv('KAFKA_TOPIC', 'market-data-stream')
    partitions = int(os.getenv('KAFKA_PARTITIONS', '3'))
    replication_factor = int(os.getenv('KAFKA_REPLICATION_FACTOR', '1'))
    
    success = create_topic(bootstrap_servers, topic_name, partitions, replication_factor)
    sys.exit(0 if success else 1)

