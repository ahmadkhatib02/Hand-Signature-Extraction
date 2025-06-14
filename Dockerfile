# Dockerfile for Hand Signature Extraction API
# This container packages the signature detection application with all dependencies
# Base image includes Python 3.9, FastAPI, Uvicorn, and Gunicorn for production deployment

# Use the official FastAPI base image with Python 3.9
# This image includes Uvicorn and Gunicorn ASGI servers pre-configured
# tiangolo/uvicorn-gunicorn-fastapi provides optimized FastAPI deployment setup
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Copy Python dependencies file to the container
# This is done early to leverage Docker layer caching for dependencies
COPY ./requirements.txt /app/requirements.txt

# Install system dependencies required for OpenCV and image processing
# ffmpeg: Video/audio processing library (required by OpenCV)
# libsm6: X11 Session Management library (required for GUI operations)
# libxext6: X11 extensions library (required for display operations)
# -y flag automatically answers 'yes' to installation prompts
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install Python dependencies from requirements.txt
# --no-cache-dir: Don't cache downloaded packages (reduces image size)
# --upgrade: Upgrade packages to latest compatible versions
# -r: Install from requirements file
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the application source code to the container
# This includes the FastAPI app and model files
# Done last to optimize Docker layer caching (code changes more frequently than dependencies)
COPY ./app /app