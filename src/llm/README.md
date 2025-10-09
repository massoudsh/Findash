# LLM Module Documentation

This document provides an overview of the Language Model (LLM) module, which is responsible for fine-tuning and inference tasks.

## Table of Contents

- [Module Architecture](#module-architecture)
- [Key Components](#key-components)
  - [`config.py`](#configpy)
  - [`training.py`](#trainingpy)
  - [`inference.py`](#inferencepy)
- [Asynchronous Fine-Tuning](#asynchronous-fine-tuning)
  - [Celery Task](#celery-task)
  - [Triggering a Job](#triggering-a-job)

## Module Architecture

The `llm` module is designed to be a self-contained unit for all language model operations. It follows a standard structure for a machine learning module:

-   **Configuration**: Centralized settings for models, training parameters, etc.
-   **Training**: Scripts and logic for fine-tuning models.
-   **Inference**: Services for running predictions with trained models.
-   **Tasks**: Asynchronous jobs (Celery) for long-running processes like fine-tuning.

## Key Components

### `config.py`

This file contains all the configuration variables for the LLM module, such as the base model name (`MODEL_NAME_OR_PATH`) and other hyperparameters. Centralizing the configuration makes it easy to manage and modify model settings without changing the core logic.

### `training.py`

This file contains the `FinetuningPipeline` class, which encapsulates the entire process of fine-tuning a model. This includes:
-   Loading the base model and tokenizer.
-   Preparing the dataset.
-   Setting up the `peft` configuration for LoRA.
-   Running the `transformers.Trainer`.
-   Saving the fine-tuned adapter.

### `inference.py`

This file provides the `InferenceService` class, which is used to perform predictions. It loads a fine-tuned model (or a base model) and provides a `process_text` method to run inference on new data. It is optimized for use within the FastAPI application.

## Asynchronous Fine-Tuning

### Celery Task

To avoid blocking the main API thread, the fine-tuning process is handled by a Celery task located in `src/llm/tasks/finetuning_task.py`. This task wraps the `FinetuningPipeline` and can be executed asynchronously by a Celery worker.

### Triggering a Job

A fine-tuning job can be started by making a `POST` request to the `/llm/finetune` API endpoint. This endpoint triggers the Celery task and immediately returns a `task_id`, which can be used to monitor the job's status via the `/llm/finetune/{task_id}` endpoint. 