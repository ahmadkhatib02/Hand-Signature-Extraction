# Hand Signature Extraction (fine-tuned YOLO v8n)

## Project Overview

This project implements a fine-tuned YOLO v8 nano model for detecting and cropping hand signatures from images. The model is trained on multiple hand-signature datasets, containing diverse hand signature samples, allowing it to accurately detect and extract signatures from complex backgrounds, deployed as a FastAPI application, and containerized with Docker.

### Key Features

- Fine-tuned YOLO v8 nano model for hand signature detection
- FastAPI-based REST API for easy integration
- Docker containerization for simplified deployment
- Automatic signature cropping and saving

## Setup and Installation

1. Build the Docker image:

   ```
   docker build -t Hand-Signature-Extraction .
   ```

2. Run the Docker container:
   ```
   docker run -p 8000:8000 Hand-Signature-Extraction
   ```

The API will now be available at `http://localhost:8000`.

## Usage

To use the hand signature detection model:

1. Ensure the Docker container is running.
2. Send a POST request to the `/predict` endpoint with an image file.

## API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Signature detection and cropping endpoint
  - Input: Image file
  - Output: Cropped signature image
