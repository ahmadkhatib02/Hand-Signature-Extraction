"""
Hand Signature Extraction API

This FastAPI application provides endpoints for extracting signatures from uploaded images
using a fine-tuned YOLO v8 model. The API accepts image uploads and returns cropped
signature images.

Author: Hand Signature Extraction Project
Purpose: Signature detection and extraction from documents
Dependencies: FastAPI, PIL, ultralytics YOLO model
"""

# Import FastAPI framework components for building the REST API
from fastapi import FastAPI, UploadFile
# Import our custom signature cropping function from the model module
from model.model import cropSignature
# Import io for handling byte streams from uploaded files
import io
# Import PIL (Python Imaging Library) for image processing operations
from PIL import Image
# Import FileResponse for returning image files as HTTP responses
from fastapi.responses import FileResponse

# Initialize the FastAPI application instance
app = FastAPI()

@app.get("/")
def home():
    """
    Health check endpoint

    Returns:
        dict: Simple health status indicating the API is running

    Note: Could be extended to include model version information
    """
    # Basic health check response - could include model version in future
    # return {"health_check": "OK", "model_version": model_version}
    return {"health_check": "OK"}

@app.post("/predict")
def predict(image: UploadFile):
    """
    Signature prediction endpoint

    Accepts an uploaded image file, processes it through the YOLO model to detect
    and crop signatures, then returns the cropped signature as a file response.

    Args:
        image (UploadFile): The uploaded image file containing potential signatures

    Returns:
        FileResponse: The cropped signature image file

    Process:
        1. Read the uploaded file into memory
        2. Convert bytes to PIL Image object
        3. Process through YOLO model with confidence threshold of 0.2
        4. Return the cropped signature as a file download
    """
    # Read the uploaded file content into memory as bytes
    input_image = image.file.read()

    # Convert the byte stream to a PIL Image object for processing
    input_image = Image.open(io.BytesIO(input_image))

    # Process the image through our signature detection model
    # confidence=0.2 means we accept detections with 20% or higher confidence
    cropped_signature = cropSignature(image=input_image, save_dir="./", confidance=0.2)

    # The cropSignature function returns: {"signature_image": signature, "signature_path": signature_path}
    # We return the file path as a downloadable response
    return FileResponse(cropped_signature["signature_path"])

    # Alternative: return the dictionary directly (commented out)
    # return cropped_signature