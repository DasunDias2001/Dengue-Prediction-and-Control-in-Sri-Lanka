"""
Mosquito Classification API
FastAPI backend for dengue mosquito species identification
"""
from __future__ import annotations

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import uvicorn

from model_handler import get_classifier
from utils import save_upload_file, cleanup_upload_file, validate_image_file

# Type aliases to avoid unused import warnings
ResponseDict = Dict[str, Any]

# Initialize FastAPI app
app = FastAPI(
    title="Mosquito Classification API",
    description="API for classifying dengue mosquito species (Aedes aegypti vs Aedes albopictus)",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    """Load model when server starts."""
    print("Starting Mosquito Classification API...")
    try:
        get_classifier()
        print(" Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")


@app.get("/")
async def health_check() -> ResponseDict:
    """Check if API and model are ready."""
    classifier = get_classifier()
    return {
        "status": "online",
        "model_loaded": classifier.model is not None
    }


@app.post("/predict")
async def predict_mosquito(file: UploadFile = File(...)) -> ResponseDict:
    """
    Predict mosquito species from uploaded image.
    
    Args:
        file: Image file (jpg, jpeg, png, bmp, tiff)
        
    Returns:
        Prediction results with class and confidence
    """
    if not validate_image_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image (jpg, jpeg, png, bmp, tiff)"
        )
    
    file_path = None
    try:
        file_path = save_upload_file(file)
        print(f"📥 Received file: {file.filename}")
        print(f"💾 Saved to: {file_path}")
        
        classifier = get_classifier()
        result = classifier.predict(file_path)
        print(f"🔮 Prediction: {result}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if file_path:
            cleanup_upload_file(file_path)


@app.get("/model-info")
async def get_model_info() -> ResponseDict:
    """Get information about the loaded model."""
    classifier = get_classifier()
    return {
        "model_path": str(classifier.model_path),
        "classes": classifier.class_names,
        "input_size": classifier.img_size,
        "model_loaded": classifier.model is not None
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )