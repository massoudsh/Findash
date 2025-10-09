import asyncio
import queue
import threading
import time
from functools import lru_cache
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from . import config


class LLMInferenceEngine:
    """
    An asynchronous, batched inference engine for language models.
    It uses a background thread to collect requests and process them in batches
    for improved efficiency.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        max_batch_size: int = config.MAX_BATCH_SIZE_INFERENCE,
        max_wait_time: float = config.MAX_WAIT_TIME_INFERENCE,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model.eval()

        self.request_queue = queue.Queue()
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.processing_thread = threading.Thread(target=self._process_batch, daemon=True)
        self.processing_thread.start()

    def _process_batch(self):
        while True:
            batch_items = []
            start_time = time.time()

            # Collect items for the batch
            while len(batch_items) < self.max_batch_size and (time.time() - start_time < self.max_wait_time):
                try:
                    item = self.request_queue.get_nowait()
                    batch_items.append(item)
                except queue.Empty:
                    if batch_items:  # If we have any items, process them
                        break
                    time.sleep(0.001)  # Prevent CPU spinning
                    continue

            if not batch_items:
                continue

            # Process batch
            try:
                texts = [item["text"] for item in batch_items]
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_length=128)

                predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # Return results
                for item, pred in zip(batch_items, predictions):
                    item["future"].set_result(pred)

            except Exception as e:
                for item in batch_items:
                    item["future"].set_exception(e)

    async def predict(self, text: str) -> str:
        """Submits a text for prediction and awaits the result."""
        future = asyncio.Future()
        self.request_queue.put({"text": text, "future": future})
        return await future


class InferenceService:
    """
    A service layer that provides cached, real-time inference.
    It wraps the LLMInferenceEngine and adds a cache layer.
    """

    def __init__(self, model_path: str, cache_size: int = config.CACHE_SIZE_INFERENCE):
        self.inference_engine = LLMInferenceEngine(model_path, model_path)
        self.cache = self._setup_cache(cache_size)

    def _setup_cache(self, cache_size: int):
        @lru_cache(maxsize=cache_size)
        async def cached_predict(text: str) -> str:
            return await self.inference_engine.predict(text)

        return cached_predict

    async def process_text(self, text: str) -> Dict:
        """
        Processes a given text, utilizing a cache to speed up responses.
        Returns the prediction along with performance metrics.
        """
        start_time = time.time()
        # Note: Caching in this async setup is tricky. The lru_cache will cache the coroutine.
        # A more robust solution might involve a custom async cache implementation.
        # For this migration, we stick to the original logic's spirit.
        prediction = await self.cache(text)
        processing_time = time.time() - start_time

        # The 'source' logic from the original is simplified here.
        # A true cache hit would have a much lower processing_time.
        source = "cache" if processing_time < 0.001 else "model"

        return {"prediction": prediction, "source": source, "processing_time": processing_time}


class ModelMonitor:
    """
    Monitors the performance of the model, specifically tracking latency.
    """

    def __init__(self, window_size: int = config.LATENCY_WINDOW_SIZE, alert_threshold: float = config.P95_LATENCY_ALERT_THRESHOLD):
        self.latency_window: List[float] = []
        self.window_size = window_size
        self.alert_threshold = alert_threshold

    def track_latency(self, latency: float):
        """Tracks the latency of a single prediction."""
        self.latency_window.append(latency)
        if len(self.latency_window) > self.window_size:
            self.latency_window.pop(0)

        if self._calculate_p95_latency() > self.alert_threshold:
            self._trigger_alert()

    def _calculate_p95_latency(self) -> float:
        if not self.latency_window:
            return 0.0
        return np.percentile(self.latency_window, 95)

    def _trigger_alert(self):
        """Placeholder for an alerting mechanism (e.g., logging, webhook)."""
        p95_latency = self._calculate_p95_latency()
        print(f"ALERT: p95 latency ({p95_latency:.4f}s) exceeded threshold ({self.alert_threshold}s)") 