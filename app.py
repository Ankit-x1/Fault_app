from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from fault_detector import FaultDetectionSystem, load_data_from_csv
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Industrial Fault Detection API", version="2.0")

# Global detector instance
detector = None


class SensorData(BaseModel):
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


class BatchSensorData(BaseModel):
    data: List[List[float]]  # Multiple sensor readings


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from fault_detector import FaultDetectionSystem, load_data_from_csv
import logging
from typing import List
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    detector = FaultDetectionSystem()
    try:
        detector.load_model("models/industrial_fault_detector.pth")
        logger.info("Fault detection system loaded and ready")
    except Exception as e:
        logger.warning(f"Could not load pre-trained model: {e}")
    yield
    logger.info("Application shutdown")


app = FastAPI(title="Industrial Fault Detection API", version="2.0", lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "operational", "model_loaded": detector is not None}


@app.post("/train")
def train_model():
    try:
        train_data = load_data_from_csv("data/normal_data.csv")
        results = detector.train(train_data, epochs=50)
        detector.save_model("models/industrial_fault_detector.pth")
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect")
def detect_fault(sensor_data: SensorData):
    try:
        data_point = [
            sensor_data.accel_x,
            sensor_data.accel_y,
            sensor_data.accel_z,
            sensor_data.gyro_x,
            sensor_data.gyro_y,
            sensor_data.gyro_z,
        ]

        result = detector.detect_realtime(data_point)

        return {
            "is_anomaly": result.is_anomaly,
            "confidence": result.confidence,
            "reconstruction_error": result.reconstruction_error,
            "statistical_score": result.statistical_score,
            "ensemble_score": result.ensemble_score,
            "sensor_importance": result.sensor_importance,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/batch")
def detect_batch(data: BatchSensorData):
    try:
        results = []
        for reading in data.data:
            if len(reading) != 6:
                raise ValueError("Each reading must have exactly 6 values")

            result = detector.detect_realtime(reading)
            results.append(
                {
                    "is_anomaly": result.is_anomaly,
                    "confidence": result.confidence,
                    "reconstruction_error": result.reconstruction_error,
                }
            )

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
