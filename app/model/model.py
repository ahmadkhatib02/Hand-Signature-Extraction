"""
YOLO-based Signature Detection and Cropping Module

This module contains the core functionality for detecting and extracting signatures
from document images using a fine-tuned YOLO v8 nano model. The model has been
trained on multiple signature datasets to achieve high accuracy in signature detection.

Model Details:
- Base Model: YOLO v8 nano (lightweight and fast)
- Training: Fine-tuned on hand signature datasets
- Purpose: Detect signature bounding boxes in documents
- Output: Cropped signature images

Author: Hand Signature Extraction Project
Dependencies: ultralytics, PIL, pathlib, io
"""

# Import YOLO from ultralytics for object detection capabilities
from ultralytics import YOLO
# Import Path for file system path operations (currently unused but good practice)
from pathlib import Path
# Import io for handling byte streams and file operations
import io
# Import PIL for image processing and manipulation
from PIL import Image

# Load the pre-trained signature detection model
# This model has been fine-tuned specifically for hand signature detection
# Model file: yolo_v8n_finetuned_hand_signatures.pt (YOLO v8 nano architecture)
inferance_model = YOLO("./model/yolo_v8n_finetuned_hand_signatures.pt")

def cropSignature(image, save_dir, model=inferance_model, confidance=0.2):
    """
    Detect and crop signatures from an input image using YOLO model

    This function takes an input image, runs it through the YOLO signature detection
    model, and returns both the cropped signature image and its file path. The function
    uses a confidence threshold to filter out low-confidence detections.

    Args:
        image (PIL.Image): Input image containing potential signatures
        save_dir (str): Directory path where cropped signatures will be saved
        model (YOLO, optional): YOLO model instance for prediction.
                               Defaults to the pre-loaded inferance_model
        confidance (float, optional): Confidence threshold for detections (0.0-1.0).
                                    Defaults to 0.2 (20% confidence)

    Returns:
        dict: Dictionary containing:
            - "signature_image": PIL Image object of the cropped signature
            - "signature_path": String path to the saved signature file

    Process:
        1. Run YOLO prediction on input image with specified confidence threshold
        2. Extract and save the highest confidence detection as a cropped image
        3. Load the saved image back into memory
        4. Return both the image object and file path

    Note:
        - The function assumes at least one signature is detected
        - Cropped images are saved as JPG format
        - The save directory structure: save_dir/signature/croppedSignature.jpg
    """
    # Run YOLO model prediction on the input image
    # conf parameter sets minimum confidence threshold for detections
    prediction = model.predict(image, conf=confidance)

    # Save the first (highest confidence) detection as a cropped image
    # save_crop automatically creates the directory structure and saves the crop
    prediction[0].save_crop(save_dir=save_dir, file_name=save_dir+"croppedSignature")

    # Construct the path to the saved cropped signature
    # YOLO save_crop creates a 'signature' subdirectory automatically
    signature_path = save_dir+"signature/"+"croppedSignature.jpg"

    # Load the saved signature image back into memory as a PIL Image object
    # This allows for further processing or direct return to the API
    signature = Image.open(io.BytesIO(open(signature_path, "rb").read()))

    # Return both the image object and file path for flexibility
    return {"signature_image":signature, "signature_path":signature_path}