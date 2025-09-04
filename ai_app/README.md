# AI Lab - Week 1

This repository contains a Django-based backend/frontend for image processing pipline. The project is organized into several apps, 
- `api`: main endpionts count/correct,
- `config`: django setup and routine,
- `records`: database classea, 
- `web`: frontend,
 
The app supports image upload, object counting, and result correction.

## Features

- Upload images and count objects using AI segmentation.
- Correct predicted counts and update records.
- Modular Django app structure.
- REST API endpoints.

## Project Structure

```
ai_app/
├── manage.py
├── db.sqlite3
├── api/
├── config/
├── records/
├── templates/
├── var/
│   └── media/
│       └── uploads/
└── web/
```

## Setup Instructions

### 1. Clone the repository

```sh
git clone git@git.fim.uni-passau.de:aie/ai-engineering-lab/student-projects/group-1.git
cd gourp-1/ai_app
```

### 2. Create and activate a virtual environment

```sh
python3 -m venv env
source env/bin/activate
```

### 3. Install dependencies

```sh
pip install -r requirements.txt
```

### 4. Apply migrations

```sh
python manage.py makemigrations
python manage.py migrate
```

### 5. Run the development server

```sh
python manage.py runserver
```

The server will start at `http://127.0.0.1:8000/`.

## Running Tests

To run unit tests for the project:

```sh
python manage.py test
```

## API Endpoints

- `POST /api/count/` — Upload image and object type for counting.
- `POST /api/correct/` — Submit corrected count for a result.

## Notes

- Uploaded images are stored in `var/media/uploads/`.
- Update `settings.py` for custom configurations as needed.


