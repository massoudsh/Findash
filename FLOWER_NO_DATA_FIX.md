# Why Flower Shows No Data - Root Cause & Solution

## Problem Summary

Flower shows no data because **Celery workers are failing to start** due to missing Python dependencies.

## Root Cause

1. **Missing Dependencies**: 
   - `psutil` (added to requirements.txt)
   - `accelerate` (still missing - used by LLM training tasks)

2. **Worker Startup Failure**:
   - Workers crash on startup when trying to import LLM tasks
   - Error: `ModuleNotFoundError: No module named 'accelerate'`
   - Workers never register with Flower

3. **No Tasks Executing**:
   - Workers aren't running → No tasks can execute
   - Flower has nothing to monitor → Shows empty

## Current Status

✅ **Fixed**:
- Added `psutil>=5.9.0` to requirements.txt
- Commented out LLM tasks from Celery includes (optional dependencies)

❌ **Still Failing**:
- Worker still crashes (likely another import issue)
- Need to rebuild container after changes

## Solution Steps

### Option 1: Add Missing Dependencies (Recommended)

Add to `requirements.txt`:
```txt
accelerate>=0.20.0  # For LLM training tasks
```

Then rebuild:
```bash
docker-compose -f docker-compose-complete.yml build celery-worker
docker-compose -f docker-compose-complete.yml up -d celery-worker celery-beat
```

### Option 2: Make LLM Tasks Truly Optional

Ensure no other modules import LLM tasks. Check:
- `src/training/tasks.py`
- `src/analytics/service.py`
- Any other files that might import LLM modules

### Option 3: Quick Test - Use Only Core Tasks

Temporarily disable all optional tasks to get workers running:

```python
# In src/core/celery_app.py
include=[
    "src.data_processing.tasks",
    "src.data_processing.market_data_tasks",
    # Comment out all others temporarily
]
```

## Verify Workers Are Running

```bash
# Check worker status
docker logs octopus-celery-worker --tail 20

# Should see: "celery@hostname ready"
# Should NOT see: "ModuleNotFoundError"

# Check registered tasks
docker exec octopus-celery-worker \
  celery -A src.core.celery_app inspect registered

# Check Flower sees workers
curl http://localhost:5555/api/workers
# Should return: {"worker_name": {...}, ...}
```

## Generate Test Data in Flower

Once workers are running:

```python
# From Python shell or API
from src.data_processing.market_data_tasks import fetch_single_market_data

# Trigger a task
task = fetch_single_market_data.delay('AAPL')
print(f"Task ID: {task.id}")

# Check in Flower: http://localhost:5555
# Go to Tasks tab → Should see the task
```

## Expected Flower Data

Once workers are running, Flower should show:

1. **Workers Tab**:
   - Active worker(s) listed
   - Worker status, uptime, pool type

2. **Tasks Tab**:
   - Task history
   - Task states (SUCCESS, FAILURE, PENDING)
   - Task execution times

3. **Monitor Tab**:
   - Real-time task execution
   - Task rate graphs
   - Worker activity

4. **Broker Tab**:
   - Queue lengths
   - Exchange information

## Quick Fix Command

```bash
# Rebuild with all fixes
docker-compose -f docker-compose-complete.yml build celery-worker celery-beat

# Restart services
docker-compose -f docker-compose-complete.yml up -d celery-worker celery-beat

# Wait for startup
sleep 10

# Check status
docker logs octopus-celery-worker --tail 30

# Verify Flower
curl http://localhost:5555/api/workers
```

## Summary

**Flower has no data because workers aren't running.**

Fix the worker startup errors, and Flower will automatically show:
- Active workers
- Task execution history
- Real-time monitoring data

The issue is **not with Flower** - it's with the Celery workers failing to start.
