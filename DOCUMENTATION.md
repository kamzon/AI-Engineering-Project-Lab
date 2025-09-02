# System Architecture Documentation

This document describes the architecture of the **Django-based Object Counting System**.  
The system allows users to upload images, detects and counts objects using an ML pipeline, and stores results for correction and retrieval.

---

## Overview

The architecture consists of four layers:

1. **Frontend (Django Templates)**
2. **Backend (Django Views/REST API)**
3. **ML Pipeline (Python package `pipeline/`)**
4. **Database & Storage**

Data flow: **User → Frontend → Backend → ML Pipeline → DB/Storage → Frontend**.

---

## Components

### 1. Frontend (Django Templates)

- **UI Layer**
  - Upload image
  - Select object type
  - Display results
  - Provide correction input

The frontend sends HTTP requests to the backend and displays results or feedback to the user.

---

### 2. Backend (Django Views/REST API)

- **`POST /api/count`**
  - Handles image upload
  - Invokes the ML pipeline
  - Stores results in the database
  - Returns prediction results to the frontend

- **`POST /api/correct`**
  - Handles user corrections
  - Updates the database with corrected values

- **Django ORM**
  - Provides an abstraction between Django models and the SQLite database
  - Ensures seamless CRUD operations

---

### 3. ML Pipeline (`pipeline/` package)

The ML pipeline is invoked by the backend to process uploaded images:

1. **Preprocessing**  
   - Normalization, resizing, or data augmentation for the image input

2. **Object Detection**  
   - Detects objects of the specified type within the image

3. **Counting**  
   - Counts the detected objects and returns the predicted result

---

### 4. Database & Storage

- **SQLite Database**
  - Stores prediction results in a `Results` table
  - Keeps track of corrections submitted by users

- **Image Storage (File System)**
  - Stores raw uploaded images for processing and later reference

---

## Data Flow

1. **Image Upload**
   - The user uploads an image and selects the object type from the frontend.
   - The request is sent to `POST /api/count`.

2. **Backend Processing**
   - The image is saved in the file system.
   - The ML pipeline is triggered to process the image.
   - Predicted count is returned to the backend.

3. **Results Storage**
   - The backend saves results in the database via the Django ORM.
   - Prediction results are returned to the frontend.

4. **User Correction**
   - If the user submits a correction, the frontend sends it to `POST /api/correct`.
   - The database record is updated accordingly.

