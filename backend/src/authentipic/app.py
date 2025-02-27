import io
import logging
import torch
import base64
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from typing import Optional, Dict, Any, List

from authentipic.models.model_factory import ModelFactory
from authentipic.config import config
from authentipic.inference.predictor import Predictor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("authentipic.api")

app = FastAPI(
    title="AuthentiPic API", description="API for AI-generated image detection"
)

# Set up CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and predictor
model = None
predictor = None


def get_device():
    """Get the appropriate device for inference."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model():
    """Load the model for inference."""
    global model, predictor

    if model is None:
        try:
            device = get_device()
            logger.info(f"Loading model on {device}")

            # Create model
            model_config = config.model
            model = ModelFactory.get_model(model_config)

            # Load trained weights
            checkpoint_path = config.inference.best_model_path
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()

            # Create predictor
            predictor = Predictor(model, device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail="Error loading model")


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    load_model()


@app.get("/")
async def root():
    """Root endpoint to check if API is running."""
    return {"message": "AuthentiPic API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an uploaded image to detect if it's AI-generated.

    Args:
        file: Uploaded image file

    Returns:
        Analysis results including probability of being AI-generated
    """
    if model is None:
        load_model()

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess image
        from torchvision import transforms

        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image_tensor = preprocess(image).unsqueeze(0)

        # Make prediction
        prediction = predictor.predict(image_tensor)
        prob_ai_generated = float(prediction[0][1].item())

        # Results
        results = {
            "probability_ai_generated": prob_ai_generated,
            "is_ai_generated": prob_ai_generated > config.inference.threshold,
            "confidence": "high"
            if abs(prob_ai_generated - 0.5) > 0.3
            else "medium"
            if abs(prob_ai_generated - 0.5) > 0.15
            else "low",
        }

        return JSONResponse(content=results)

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/analyze/base64/")
async def analyze_image_base64(image_data: Dict[str, str]):
    """
    Analyze a base64 encoded image to detect if it's AI-generated.

    Args:
        image_data: Dictionary with base64 encoded image

    Returns:
        Analysis results including probability of being AI-generated
    """
    if model is None:
        load_model()

    try:
        # Decode base64 image
        base64_image = image_data.get("image", "")
        if not base64_image:
            raise HTTPException(status_code=400, detail="No image data provided")

        # Remove data URL prefix if present
        if "base64," in base64_image:
            base64_image = base64_image.split("base64,")[1]

        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess image
        from torchvision import transforms

        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image_tensor = preprocess(image).unsqueeze(0)

        # Make prediction
        prediction = predictor.predict(image_tensor)
        prob_ai_generated = float(prediction[0][1].item())

        # Results
        results = {
            "probability_ai_generated": prob_ai_generated,
            "is_ai_generated": prob_ai_generated > config.inference.threshold,
            "confidence": "high"
            if abs(prob_ai_generated - 0.5) > 0.3
            else "medium"
            if abs(prob_ai_generated - 0.5) > 0.15
            else "low",
        }

        return JSONResponse(content=results)

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("authentipic.api:app", host="0.0.0.0", port=8000, reload=True)
