# API Documentation

This document provides an overview of the FastAPI backend API for the Quantum Trading Matrix platform.

## Table of Contents

- [API Structure](#api-structure)
- [Adding New Endpoints](#adding-new-endpoints)
- [Available Routers](#available-routers)
  - [LLM Service](#llm-service)
  - [Portfolio API](#portfolio-api)

## API Structure

The API is built using FastAPI and is organized into a modular structure using `APIRouter`. Each distinct functional area of the application (e.g., LLM, Portfolio, Risk) has its own router file located in `src/api/endpoints/`.

All routers are imported and registered in the main `src/main.py` file. This keeps the main application file clean and delegates route management to the specific router modules.

## Adding New Endpoints

To add a new set of endpoints:

1.  **Create a new router file**: In the `src/api/endpoints/` directory, create a new Python file (e.g., `my_new_router.py`).
2.  **Define the router**: Inside the new file, create an `APIRouter` instance:
    ```python
    from fastapi import APIRouter
    my_new_router = APIRouter()
    ```
3.  **Create your endpoints**: Define your path operations using the new router instance:
    ```python
    @my_new_router.get("/hello")
    def say_hello():
        return {"message": "Hello, World!"}
    ```
4.  **Register the router**: In `src/main.py`, import your new router and include it in the main FastAPI app:
    ```python
    from src.api.endpoints.my_new_router import my_new_router
    app.include_router(my_new_router, prefix="/my-new-route", tags=["My New Route"])
    ```

## Available Routers

### LLM Service

-   **File**: `src/api/endpoints/llm.py`
-   **Prefix**: `/llm`
-   **Description**: Manages language model operations, including fine-tuning and inference.

**Endpoints**:

-   `POST /finetune`: Triggers an asynchronous fine-tuning job.
    -   **Request Body**: `{ "output_dir": "string" }`
    -   **Response**: `{ "message": "Fine-tuning job started.", "task_id": "string" }`
-   `GET /finetune/{task_id}`: Retrieves the status of a fine-tuning job.
    -   **Response**: `{ "task_id": "string", "status": "string", "result": "any" }`
-   `GET /finetune`: (Mock) Lists all fine-tuning jobs.

### Portfolio API

-   **File**: `src/api/endpoints/portfolio_api.py`
-   **Prefix**: `/portfolio`
-   **Description**: Provides access to portfolio and position data. Note: This currently serves mock data and is not connected to a database.

**Endpoints**:

-   `GET /`: Retrieves a list of all portfolios.
-   `GET /{portfolio_id}/positions`: Retrieves all positions for a specific portfolio. 