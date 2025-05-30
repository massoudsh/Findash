import asyncio
import json
import redis
from channels.generic.websocket import AsyncWebsocketConsumer

class LiveDashboardConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.group_name = "live_dashboard"
        self.redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

        # Subscribe to Redis Channel
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe("dashboard_updates")
        asyncio.create_task(self.listen_to_redis())

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)
        self.pubsub.close()

    async def receive(self, text_data):
        data = json.loads(text_data)
        await self.send(json.dumps({"message": "Received"}))

    async def listen_to_redis(self):
        for message in self.pubsub.listen():
            if message["type"] == "message":
                await self.send(json.dumps({"type": "update", "data": message["data"].decode("utf-8")}))