import os
import sys
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
import numpy as np
import logging

# Import config (moved to top)
import config

# Define Pydantic models for request bodies
class SensorReadings2D(BaseModel):
    sensor_readings: list[list[float]] = Field(..., description=f"A list of lists of sensor data points (2D array). Must contain exactly {config.SEQUENCE_LENGTH} samples.")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fault Detection API",
    description="API for training an autoencoder model and detecting anomalies in sensor data.",
    version="1.0.0"
)

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import config
import config

# Import existing functions
from train import train as train_model
from detect import detect as detect_faults

# Setup Jinja2Templates for HTML rendering
templates = Jinja2Templates(directory="templates")

# Ensure the models directory exists
os.makedirs(config.MODELS_DIR, exist_ok=True)

@app.get("/health", response_model=dict, summary="Health check endpoint")
def health_check():
    """
    Returns a simple status to indicate the application is running.
    """
    return {"status": "UP"}

@app.get("/", response_class=HTMLResponse, summary="Serve the main application page")
async def read_root(request: Request):
    """
    Serves the main `index.html` page, providing configuration details
    relevant to the sensor data and model.
    """
    logger.info("Serving index.html")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "SEQUENCE_LENGTH": config.SEQUENCE_LENGTH,
            "N_FEATURES": config.N_FEATURES,
            "NUM_EPOCHS": config.NUM_EPOCHS,
            "SENSOR_FEATURES_STRING": ", ".join(config.SENSOR_FEATURES)
        }
    )

@app.post("/api/train", response_model=dict, summary="Trigger model training")
def train_api():
    """
    Triggers the training process for the autoencoder model.
    This operation can be time-consuming.
    """
    logger.info("API call received: /api/train")
    try:
        log_message = run_in_threadpool(train_model)
        logger.info("Model training completed successfully.")
        return JSONResponse(content={"status": "success", "message": "Model training completed.", "log": log_message},
                            status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error during training API call: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during model training: {str(e)}"
        )

@app.post("/api/detect", response_model=dict, summary="Detect anomalies in sensor data")
def detect_api(
    request_body: SensorReadings2D # Only accept 2D arrays now
):
    """
    Detects anomalies in provided sensor readings.
    Input must be a 2D list of sensor data points representing a full sequence
    of `SEQUENCE_LENGTH` samples.
    """
    logger.info("API call received: /api/detect")
    try:
        sensor_readings_np = np.array(request_body.sensor_readings)

        # Validate that the input array has the correct number of samples
        if sensor_readings_np.shape[0] != config.SEQUENCE_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"'sensor_readings' must contain exactly {config.SEQUENCE_LENGTH} samples."
            )

        if sensor_readings_np.ndim != 2: # Should be 2 due to Pydantic model, but an extra check
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid format for 'sensor_readings'. Expected a 2D array-like structure."
            )
        
        # Further validation for inner list elements
        if not all(isinstance(row, list) and all(isinstance(x, (int, float)) for x in row) for row in request_body.sensor_readings):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="For 'sensor_readings', all elements must be lists of numbers."
            )
        
        results = run_in_threadpool(detect_faults, data_stream_buffer=sensor_readings_np)
        
        logger.info(f"Detection API call results: {results.get('status', 'N/A')}")
        return JSONResponse(content={"status": results["status"], "message": results.get("message", "No message provided."), "results": {
            "anomaly_threshold": results.get("anomaly_threshold"),
            "number_of_anomalies": results.get("number_of_anomalies"),
            "anomaly_indices": results.get("anomaly_indices")
        }}, status_code=status.HTTP_200_OK)
    except HTTPException as http_e:
        raise http_e
    except Exception as e:
        logger.error(f"Error during detection API call: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during fault detection: {str(e)}"
        )

if __name__ == "__main__":
    logger.info("Starting FastAPI application with Uvicorn.")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)