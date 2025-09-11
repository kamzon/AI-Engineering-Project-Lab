# Django-based Object Counting System (Group 1)

| Name              | Email                      | FIM ID |
| ----------------- | -------------------------- | ------ |
| Haidar Chaito     | chaito01@ads.uni-passau.de | chaito |
| Noureddine Kamzon | kamzon01@ads.uni-passau.de | kamzon |

## Overview
This app lets users upload an image and a target object type. The backend runs a modular ML pipeline to segment objects, classify them, and return per-class counts. Users can correct results, and the system exposes rich metrics to track performance.

Key components:
- Django backend (REST endpoints) and simple template-based UI
- ML pipeline (`pipeline/`): SAM-based segmentation + ResNet classifier + optional zero-shot labeling
- Database: SQLite for results and corrections
- Monitoring: Prometheus metrics + Grafana dashboards

Repository docs:
- High-level architecture: `DOCUMENTATION.md`
- Setup guide (legacy): `SETUP.md`
- API reference: `docs/API.md`

## Prerequisites
- Python 3.12
- macOS/Linux/WSL recommended
- Optional: Docker (Prometheus + Grafana)

## Quickstart
1) Create a virtual environment and install dependencies
```sh
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

2) Run migrations and start the server
```sh
python ai_app/manage.py makemigrations
python ai_app/manage.py migrate
python ai_app/manage.py runserver
```
Server: http://127.0.0.1:8000/

3) Open the web UI
- Go to http://127.0.0.1:8000/ and upload an image for counting.

## API (short)
Main endpoints (see full details in `docs/API.md`):
- `POST /api/count/` — Upload image + object_type; returns counts and metadata.
- `POST /api/correct/` — Submit a corrected count for a given result id.
- Metrics: `/prometheus/metrics` (Prometheus scrape path).

## Monitoring (Prometheus + Grafana)
We include a Docker Compose stack that provisions:
- Prometheus (scrapes the Django app at `host.docker.internal:8000`)
- Grafana (pre-provisioned dashboards and Prometheus datasource)

Start the stack (in a separate terminal):
```sh
docker-compose up
```

Open Grafana: http://localhost:3000 (admin / admin)

Dashboard: AI Lab → "Pipeline Overview". Panels include:
- Pipeline runs (total)
- Inference timings (SAM, classifier, zero-shot, overall)
- Objects by type
- Image/segment health

## Demo: Image Generation Script
Use `image_generation.py` to generate images via an external API and post them to `/api/count/`, then simulate corrections to `/api/correct/`.

Example flow:
```sh
source env/bin/activate
python image_generation.py
```
Watch Grafana for metrics changes (runs, timings, object distribution).

## ML Notebooks (Few-shot & Segmentation)
- `model_improvement_fewshot.ipynb`: prepare data, fine-tune, evaluate, visualize.
- `segmentation_and_few_shot_notebook.ipynb`: segmentation overlays, few-shot classification, counting.

## Project Structure (excerpt)
```
group-1/
├── ai_app/            # Django project & apps (api, web, records)
├── pipeline/          # ML pipeline, models, utils, outputs
├── config/            # Prometheus/Grafana configs
├── presentations/     # Slides & images
├── playground/        # Sample images & notebooks
└── docs/              # API docs and references
```

## Troubleshooting
- Docker metrics stack fails to start: ensure Docker Desktop is running and ports 9090/3000 are free.
- Prometheus shows target down: verify Django is running and `/prometheus/metrics` is reachable.
- Notebook GPU errors: set device to CPU if CUDA not available.

## License
Academic use for AI Engineering Lab exercises.
