# Containerized Audience Segmentation & Personalization Service

> **Hackathon Submission:** IT HAPPENS @ RAALE  
> **Track:** Containerized Audience Segmentation & Personalization Service  
> **Status:** Validated, Production-Ready Microservices Architecture  

An end-to-end, CPU-friendly, decoupled audience intelligence and personalization system for Over-The-Top (OTT) streaming platforms. It performs automated data validation, behavioral feature engineering, unsupervised clustering (K-Means), empirical cluster profiling, real-time REST API inference, transparent rule-based content recommendations, and independent integration evaluation.

---

## Architecture Overview

The system implements a strictly decoupled three-service architecture orchestrated via Docker Compose:

```
                  ┌──────────────────────┐
                  │ data/user_activity   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │       TRAINER        │
                  │                      │
                  │ - Data Validation    │
                  │ - Cleaning & Impute  │
                  │ - Feature Eng.       │
                  │ - Scaling            │
                  │ - K-Means (K=2..6)   │
                  │ - Profiling & Naming │
                  └──────────┬───────────┘
                             │ writes
                             ▼
                  ┌──────────────────────┐
                  │    /models volume    │
                  │  model_bundle.joblib │
                  └──────────┬───────────┘
                             │ reads
                             ▼
                  ┌──────────────────────┐
                  │         API          │
                  │                      │
                  │ - GET /health        │
                  │ - POST /recommend    │
                  │ - Validation & Guard │
                  │ - Centroid Distance  │
                  │ - Rule Recommender   │
                  └──────────┬───────────┘
                             │ HTTP
                             ▼
                  ┌──────────────────────┐
                  │      EVALUATOR       │
                  │                      │
                  │ - Wait for Health    │
                  │ - API Contract Tests │
                  │ - Edge Case Suite    │
                  │ - ML Quality Checks  │
                  │ - results/metrics.json│
                  └──────────────────────┘
```

---

## Quick Start: Run Everything in One Command

To build and run the entire system with Docker Compose:

```bash
docker compose up --build
```

### What Happens Automatically:
1. **`trainer`** starts, validates the dataset, trains the K-Means clustering model ($K=5$, Silhouette: 0.4206), profiles the segments, and writes the model bundle to the shared Docker volume (`model_volume`).
2. **`api`** starts after `trainer` completes, loads the model artifact into memory, exposes port `5000`, and passes its container healthcheck.
3. **`evaluator`** waits for `api` health, runs representative profile tests and edge-case suites, evaluates clustering quality, and generates `results/metrics.json`.

---

## Prerequisites & Docker Installation

### System Prerequisites
- **OS:** Windows 10/11, macOS, or Linux
- **Hardware:** Commodity x86_64 / ARM CPU, 2GB+ RAM available
- **Network:** Internet access during initial image build (pip packages), 100% offline thereafter

### Docker Installation Guide
1. **Windows & macOS:**
   - Download and install **Docker Desktop** from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/).
   - On Windows, ensure WSL 2 backend is enabled during installation.
   - Start Docker Desktop and ensure the engine status shows "Engine running".
2. **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io docker-compose-plugin
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER
   ```
3. **Verify Installation:**
   ```bash
   docker --version
   docker compose version
   ```

---

## Project Structure

```text
Containerized Audience Segmentation & Personalization Service/
├── data/
│   └── user_activity.csv            # 5,005 OTT viewer records
├── trainer/
│   ├── Dockerfile                   # Lean Python 3.11-slim container
│   ├── requirements.txt             # Pinned dependencies
│   └── train.py                     # Cleaning, scaling, K-Means, profiling, bundle export
├── api/
│   ├── Dockerfile                   # Container with built-in healthcheck
│   ├── requirements.txt             # Pinned dependencies
│   ├── app.py                       # REST API (GET /health, POST /recommend)
│   ├── schemas.py                   # Input validation and defensive guards
│   ├── recommender.py               # Transparent rule-based recommendation logic
│   └── model_loader.py              # Thread-safe artifact loader with caching
├── evaluator/
│   ├── Dockerfile                   # Independent test harness container
│   ├── requirements.txt             # Pinned test dependencies
│   ├── evaluate.py                  # Health waiter, API tester, metrics compiler
│   └── test_cases.json              # Valid profiles & 10 distinct edge-case fixtures
├── models/
│   ├── model_bundle.joblib          # Persisted model, scaler, and segment metadata
│   └── training_metrics.json        # Pre-computed model metrics cache
├── results/
│   └── metrics.json                 # Machine-readable evaluation report
├── docker-compose.yml               # 3-service orchestration with health dependency
├── .dockerignore                    # Build optimization rules
├── REPORT.md                        # Comprehensive 21-section technical evaluation
└── README.md                        # Quick-start guide and documentation
```

---

## Local Native Execution (Without Docker)

If running directly on a machine with Python 3.10+:

```bash
# 1. Install dependencies
pip install -r trainer/requirements.txt
pip install -r api/requirements.txt
pip install -r evaluator/requirements.txt

# 2. Train the model
python trainer/train.py

# 3. Start the API server
python api/app.py

# 4. In another terminal, run the independent evaluator
python evaluator/evaluate.py
```

---

## API Documentation

### 1. Interactive Web Dashboard (UI)
- **URL:** `GET /` (e.g. [`http://localhost:5000`](http://localhost:5000))
- **Description:** A browser interface with sliders, quick-fill viewer archetypes, live prediction, cluster breakdown, and system health monitoring.

### 2. Health Endpoint
- **URL:** `GET /health`
- **Response (200 OK):**
  ```json
  {
    "status": "ok",
    "model_loaded": true
  }
  ```

### 3. Recommendation Endpoint
- **URL:** `POST /recommend`
- **Headers:** `Content-Type: application/json`
- **Sample Request:**
  ```json
  {
    "user_id": "USR-8192",
    "watch_time_hours": 55.0,
    "top_genres": ["Action", "Thriller"],
    "avg_session_mins": 90.0
  }
  ```
- **Sample Response (200 OK):**
  ```json
  {
    "user_id": "USR-8192",
    "segment_id": 1,
    "segment_name": "High-Engagement Action Viewers",
    "recommendations": [
      "The Grand Heist",
      "Cyberstrike: Protocol Zero",
      "Crown & Treason"
    ],
    "distance_to_centroid": 0.1899
  }
  ```

### Curl Test Commands

```bash
# Health Check
curl http://localhost:5000/health

# High Engagement Action Recommendation
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id":"USR-8192","watch_time_hours":55.0,"avg_session_mins":90.0,"top_genres":["Action","Thriller"]}'

# Casual Short-Session Recommendation
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id":"USR-1024","watch_time_hours":12.0,"avg_session_mins":25.0,"top_genres":["Comedy","Romance"]}'
```

---

## Clusters & Segment Profiles

Based on empirical mathematical evidence ($K=5$ peaked at Silhouette Score **0.4206**):

| ID | Segment Name | Share | Mean Watch Time | Mean Session | Characteristics |
| :-: | :--- | :-: | :-: | :-: | :--- |
| **0** | **Eclectic Multi-Genre Explorers** | 13.8% | 47.7h | 64.1m | Diversity = 4.04; streams across 5+ genres. |
| **1** | **High-Engagement Action Viewers** | 25.7% | 54.9h | 89.7m | Heavy viewers; intense preference for Action/Thriller. |
| **2** | **Casual Short-Session Viewers** | 30.1% | 12.0h | 25.0m | Quick bite-sized comedy, animation, and romance. |
| **3** | **Low-Activity Dormant Viewers** | 10.1% | 3.5h | 18.6m | Inactive viewers (>38 days recency), minimal viewing. |
| **4** | **Weekend Binge Enthusiasts** | 20.3% | 39.6h | 119.2m | 80% viewing on weekends; marathon drama/sci-fi sessions. |

---

## Evaluator & Machine-Readable Metrics

The independent evaluator runs automatically and records live metrics to `results/metrics.json`:

```json
{
  "model": {
    "algorithm": "KMeans",
    "n_clusters": 5,
    "random_state": 42
  },
  "clustering": {
    "silhouette_score": 0.4206,
    "inertia": 10330.4,
    "davies_bouldin": 1.0334,
    "calinski_harabasz": 3215.35,
    "cluster_sizes": {
      "2": 1502,
      "1": 1283,
      "4": 1013,
      "0": 689,
      "3": 505
    }
  },
  "api": {
    "health_check": true,
    "valid_requests_passed": 4,
    "total_valid_requests": 4,
    "invalid_requests_handled": 6,
    "total_invalid_requests": 6
  },
  "edge_cases": {
    "unknown_genre": true,
    "empty_genres": true,
    "zero_watch_time": true,
    "very_large_watch_time": true,
    "very_large_session_duration": true,
    "missing_required_field": true,
    "string_instead_of_numeric": true,
    "negative_numeric_value": true,
    "repeated_request_deterministic": true,
    "malformed_json": true,
    "model_not_ready_handled_503": true
  },
  "reproducibility": {
    "fixed_seed": true,
    "seed_value": 42,
    "artifact_exists": true
  }
}
```

---

## Technical Report

For the in-depth 21-section technical evaluation covering feature selection, elbow curves, hyperparameter analysis, failure modes, and engineering decisions, see [REPORT.md](REPORT.md).

---

## Troubleshooting

- **Port 5000 Already in Use:** If another service occupies port 5000, change `"5000:5000"` to `"5001:5000"` in `docker-compose.yml`.
- **Permission Issues in Docker Volume:** The Dockerfiles create a non-root `appuser` (UID 1000) and initialize `/app/models` with appropriate permissions.
- **Model Missing on Fresh Boot:** Docker Compose uses `condition: service_completed_successfully` for `trainer`, guaranteeing the API will not boot until `model_bundle.joblib` exists on disk.
