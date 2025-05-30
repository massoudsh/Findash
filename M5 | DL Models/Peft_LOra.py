import gc
import os
import sys
import threading
import queue
from typing import List, Dict
import asyncio
import time
from functools import lru_cache

import psutil
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from peft import LoraConfig, TaskType, get_peft_model

import numpy as np


def levenshtein_distance(str1, str2):
    # TC: O(N^2)
    # SC: O(N)
    if str1 == str2:
        return 0
    num_rows = len(str1) + 1
    num_cols = len(str2) + 1
    dp_matrix = list(range(num_cols))
    for i in range(1, num_rows):
        prev = dp_matrix[0]
        dp_matrix[0] = i
        for j in range(1, num_cols):
            temp = dp_matrix[j]
            if str1[i - 1] == str2[j - 1]:
                dp_matrix[j] = prev
            else:
                dp_matrix[j] = min(prev, dp_matrix[j], dp_matrix[j - 1]) + 1
            prev = temp
    return dp_matrix[num_cols - 1]


def get_closest_label(eval_pred, classes):
    min_id = sys.maxsize
    min_edit_distance = sys.maxsize
    for i, class_label in enumerate(classes):
        edit_distance = levenshtein_distance(eval_pred.strip(), class_label)
        if edit_distance < min_edit_distance:
            min_id = i
            min_edit_distance = edit_distance
    return classes[min_id]


# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


class RealTimeInferenceEngine:
    def __init__(self, model_path: str, tokenizer_path: str, max_batch_size: int = 32, max_wait_time: float = 0.1):
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
            while len(batch_items) < self.max_batch_size and time.time() - start_time < self.max_wait_time:
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
                texts = [item['text'] for item in batch_items]
                inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_length=128)
                
                predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Return results
                for item, pred in zip(batch_items, predictions):
                    item['future'].set_result(pred)
            
            except Exception as e:
                for item in batch_items:
                    item['future'].set_exception(e)

    async def predict(self, text: str) -> str:
        future = asyncio.Future()
        self.request_queue.put({'text': text, 'future': future})
        return await future


class FinancialModelService:
    def __init__(self, model_path: str, cache_size: int = 1000):
        self.inference_engine = RealTimeInferenceEngine(model_path)
        self.cache = self._setup_cache(cache_size)
    
    def _setup_cache(self, cache_size: int):
        @lru_cache(maxsize=cache_size)
        def cache_func(text: str) -> str:
            return text  # Will be replaced with actual prediction
        return cache_func

    async def process_transaction(self, 
                                transaction_text: str, 
                                confidence_threshold: float = 0.8) -> Dict:
        # Check cache first
        cached_result = self.cache(transaction_text)
        if cached_result != transaction_text:  # Cache hit
            return {
                'prediction': cached_result,
                'source': 'cache',
                'processing_time': 0
            }

        # Real-time processing
        start_time = time.time()
        prediction = await self.inference_engine.predict(transaction_text)
        processing_time = time.time() - start_time

        # Update cache
        self.cache(transaction_text)

        return {
            'prediction': prediction,
            'source': 'model',
            'processing_time': processing_time
        }


class FinancialModelMonitor:
    def __init__(self):
        self.latency_window = []
        self.window_size = 1000
        self.alert_threshold = 0.1  # 100ms

    def track_latency(self, latency: float):
        self.latency_window.append(latency)
        if len(self.latency_window) > self.window_size:
            self.latency_window.pop(0)
        
        # Check for performance degradation
        if self._calculate_p95_latency() > self.alert_threshold:
            self._trigger_alert()

    def _calculate_p95_latency(self):
        return np.percentile(self.latency_window, 95)

    def _trigger_alert(self):
        # Implement your alerting logic here
        pass


async def main():
    accelerator = Accelerator()
    # model_name_or_path = "bigscience/T0_3B"
    model_name_or_path = "facebook/bart-large"
    dataset_name = "twitter_complaints"
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )
    text_column = "Tweet text"
    label_column = "text_label"
    lr = 3e-3
    num_epochs = 5
    batch_size = 8
    seed = 42
    do_test = False
    set_seed(seed)

    dataset = load_dataset("ought/raft", dataset_name)
    classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
    dataset = dataset.map(
        lambda x: {"text_label": [classes[label] for label in x["Label"]]},
        batched=True,
        num_proc=1,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[label_column]
        model_inputs = tokenizer(inputs, truncation=True)
        labels = tokenizer(
            targets, max_length=target_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["train"]
    test_dataset = processed_datasets["test"]

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)

    # creating model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler
    )
    accelerator.print(model)

    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    for epoch in range(num_epochs):
        with TorchTracemalloc() as tracemalloc:
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print(f"GPU Memory before entering the train : {b2mb(tracemalloc.begin)}")
        accelerator.print(f"GPU Memory consumed at the end of the train (end-begin): {tracemalloc.used}")
        accelerator.print(f"GPU Peak Memory consumed during the train (max-begin): {tracemalloc.peaked}")
        accelerator.print(
            f"GPU Total Peak Memory consumed during the train (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)}"
        )

        accelerator.print(f"CPU Memory before entering the train : {b2mb(tracemalloc.cpu_begin)}")
        accelerator.print(f"CPU Memory consumed at the end of the train (end-begin): {tracemalloc.cpu_used}")
        accelerator.print(f"CPU Peak Memory consumed during the train (max-begin): {tracemalloc.cpu_peaked}")
        accelerator.print(
            f"CPU Total Peak Memory consumed during the train (max): {tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)}"
        )
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

        model.eval()
        eval_preds = []
        with TorchTracemalloc() as tracemalloc:
            for _, batch in enumerate(tqdm(eval_dataloader)):
                batch = {k: v for k, v in batch.items() if k != "labels"}
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model).generate(
                        **batch, synced_gpus=is_ds_zero_3
                    )  # synced_gpus=True for DS-stage 3
                outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
                preds = accelerator.gather_for_metrics(outputs).detach().cpu().numpy()
                eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print(f"GPU Memory before entering the eval : {b2mb(tracemalloc.begin)}")
        accelerator.print(f"GPU Memory consumed at the end of the eval (end-begin): {tracemalloc.used}")
        accelerator.print(f"GPU Peak Memory consumed during the eval (max-begin): {tracemalloc.peaked}")
        accelerator.print(
            f"GPU Total Peak Memory consumed during the eval (max): {tracemalloc.peaked + b2mb(tracemalloc.begin)}"
        )

        accelerator.print(f"CPU Memory before entering the eval : {b2mb(tracemalloc.cpu_begin)}")
        accelerator.print(f"CPU Memory consumed at the end of the eval (end-begin): {tracemalloc.cpu_used}")
        accelerator.print(f"CPU Peak Memory consumed during the eval (max-begin): {tracemalloc.cpu_peaked}")
        accelerator.print(
            f"CPU Total Peak Memory consumed during the eval (max): {tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)}"
        )

        correct = 0
        total = 0
        assert len(eval_preds) == len(
            dataset["train"][label_column]
        ), f"{len(eval_preds)} != {len(dataset['train'][label_column])}"
        for pred, true in zip(eval_preds, dataset["train"][label_column]):
            if pred.strip() == true.strip():
                correct += 1
            total += 1
        accuracy = correct / total * 100
        accelerator.print(f"{accuracy=}")
        accelerator.print(f"{eval_preds[:10]=}")
        accelerator.print(f"{dataset['train'][label_column][:10]=}")

    if do_test:
        model.eval()
        test_preds = []
        for _, batch in enumerate(tqdm(test_dataloader)):
            batch = {k: v for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = accelerator.unwrap_model(model).generate(
                    **batch, synced_gpus=is_ds_zero_3
                )  # synced_gpus=True for DS-stage 3
            outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
            preds = accelerator.gather(outputs).detach().cpu().numpy()
            test_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        test_preds_cleaned = []
        for _, pred in enumerate(test_preds):
            test_preds_cleaned.append(get_closest_label(pred, classes))

        test_df = dataset["test"].to_pandas()
        assert len(test_preds_cleaned) == len(test_df), f"{len(test_preds_cleaned)} != {len(test_df)}"
        test_df[label_column] = test_preds_cleaned
        test_df["text_labels_orig"] = test_preds
        accelerator.print(test_df[[text_column, label_column]].sample(20))

        pred_df = test_df[["ID", label_column]]
        pred_df.columns = ["ID", "Label"]

        os.makedirs(f"data/{dataset_name}", exist_ok=True)
        pred_df.to_csv(f"data/{dataset_name}/predictions.csv", index=False)

    accelerator.wait_for_everyone()
    # Option1: Pushing the model to Hugging Face Hub
    # model.push_to_hub(
    #     f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}".replace("/", "_"),
    #     token = "hf_..."
    # )
    # token (`bool` or `str`, *optional*):
    #     `token` is to be used for HTTP Bearer authorization when accessing remote files. If `True`, will use the token generated
    #     when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
    #     is not specified.
    #     Or you can get your token from https://huggingface.co/settings/token

    # Option2: Saving the model locally
    peft_model_id = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}".replace(
        "/", "_"
    )
    model.save_pretrained(peft_model_id)
    accelerator.wait_for_everyone()

    # Initialize the real-time service
    service = FinancialModelService(peft_model_id)
    monitor = FinancialModelMonitor()

    # Example real-time processing
    async def process_stream(transactions: List[str]):
        tasks = []
        for transaction in transactions:
            task = asyncio.create_task(service.process_transaction(transaction))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results

    # Example usage
    transactions = [
        "PAYMENT *AMAZON.COM",
        "TRANSFER TO SAVINGS",
        "ATM WITHDRAWAL"
    ]
    
    results = await process_stream(transactions)
    for transaction, result in zip(transactions, results):
        print(f"Transaction: {transaction}")
        print(f"Classification: {result['prediction']}")
        print(f"Processing Time: {result['processing_time']*1000:.2f}ms")
        monitor.track_latency(result['processing_time'])


if __name__ == "__main__":
    asyncio.run(main())

