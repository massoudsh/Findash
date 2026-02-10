# Frontend Documentation

This document provides an overview of the Next.js frontend for the Octopus trading platform.

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
  - [Pages and Layouts](#pages-and-layouts)
  - [Components](#components)
- [API Communication](#api-communication)
  - [Mock Services](#mock-services)
  - [API Proxies](#api-proxies)

## Getting Started

1.  **Navigate to the frontend directory**:
    ```bash
    cd frontend-nextjs
    ```
2.  **Install dependencies**:
    ```bash
    npm install
    ```
3.  **Run the development server**:
    ```bash
    npm run dev
    ```
The application will be available at [http://localhost:3000](http://localhost:3000).

## Project Structure

The frontend code is organized into the following key directories:

-   `src/app`: Contains the pages and layouts of the application, following the Next.js App Router convention. Each subdirectory represents a route.
-   `src/components`: Contains all reusable React components. They are further organized into subdirectories based on the feature they belong to (e.g., `dashboard`, `portfolio`, `risk`).
-   `src/lib`: Contains utility functions (`utils.ts`) and API service modules (`services/`).

## Architecture

### Pages and Layouts

The application uses the Next.js App Router. Each page is defined by a `page.tsx` file within a route directory in `src/app`. These page files are kept minimal and are primarily responsible for rendering the main content component for that page, often wrapped in a `Suspense` boundary for better loading states.

### Components

The UI is built with a component-based architecture. Each page's functionality is encapsulated within a "content" component (e.g., `DashboardContent`, `PortfolioContent`). These content components are responsible for fetching data and composing smaller, reusable UI components to build the page.

## API Communication

The frontend communicates with the backend through a two-layered approach to facilitate development and maintain a clean architecture.

### Mock Services

During development, each feature area has its own mock API service file (e.g., `src/lib/services/portfolio_api.ts`). These files export asynchronous functions that return hardcoded mock data, allowing for UI development and testing without a live backend connection.

### API Proxies

For production and integration, the frontend does not call the FastAPI backend directly. Instead, it calls API routes within the Next.js application itself, located in `src/app/api/`. These routes act as proxies, forwarding requests to the actual backend.

This approach offers several advantages:
-   It hides the backend URL from the client.
-   It provides a single place to handle authentication, logging, or other middleware for API requests.
-   It avoids CORS issues during development. 