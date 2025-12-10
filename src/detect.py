import os
import sys
import torch
import numpy as np
import joblib
import onnxruntime as rt
import logging # Import logging

# Configure logging
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from data_loader import load_and_preprocess_data
from model import create_autoencoder

# Global variable to store sensor readings for real-time buffering
_sensor_reading_buffer = np.zeros((0, config.N_FEATURES))

def _get_model_and_session(device):
    logger.info("Attempting to load model (preferring ONNX).")
    model = None
    session = None
    
    # Try loading ONNX model first
    if os.path.exists(config.ONNX_MODEL_PATH):
        try:
            session = rt.InferenceSession(config.ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
            logger.info(f"Loaded ONNX model from {config.ONNX_MODEL_PATH}")
            return model, session
        except Exception as e:
            logger.warning(f"Could not load ONNX model. Falling back to PyTorch. Error: {e}")
            session = None # Reset session if ONNX fails

    # Fallback to PyTorch model if ONNX failed or not present
    if model is None:
        model = create_autoencoder().to(device)
        model.load_state_dict(torch.load(config.PYTORCH_MODEL_PATH, map_location=device))
        model.eval()
        logger.info(f"Loaded PyTorch model from {config.PYTORCH_MODEL_PATH}")
    
    return model, session


def detect(raw_data_point=None, data_stream_buffer=None):
    logger.info("Starting anomaly detection.")
    global _sensor_reading_buffer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Load the scaler
        scaler = joblib.load(config.SCALER_PATH)
        logger.info(f"Scaler loaded from {config.SCALER_PATH}")

        # Load trained PyTorch model or ONNX session
        model, onnx_session = _get_model_and_session(device)

        # 1. Determine threshold from normal training data
        logger.info("Determining anomaly threshold from normal data.")
        X_train_for_threshold, _ = load_and_preprocess_data(file_path=config.NORMAL_DATA_PATH, scaler=scaler)

        with torch.no_grad():
            if onnx_session:
                input_name = onnx_session.get_inputs()[0].name
                output_name = onnx_session.get_outputs()[0].name
                train_input = X_train_for_threshold.astype(np.float32)
                train_reconstruction = onnx_session.run([output_name], {input_name: train_input})[0]
                train_reconstruction = torch.from_numpy(train_reconstruction)
            else:
                train_input_tensor = torch.from_numpy(X_train_for_threshold).float().to(device)
                train_reconstruction = model(train_input_tensor)
            
            train_loss = torch.mean(torch.abs(train_reconstruction.cpu() - torch.from_numpy(X_train_for_threshold).float()), dim=(1, 2)).numpy()

        threshold = np.mean(train_loss) + config.ANOMALY_THRESHOLD_MULTIPLIER * np.std(train_loss)
        logger.info(f"Calculated Anomaly threshold: {threshold:.4f}")

        # 2. Prepare data for detection
        data_to_form_sequences = None
        if data_stream_buffer is not None:
            logger.info("Processing data from provided data_stream_buffer.")
            if data_stream_buffer.shape[0] < config.SEQUENCE_LENGTH:
                return {"status": "error", "message": f"Data stream buffer must contain at least {config.SEQUENCE_LENGTH} samples."}
            data_to_form_sequences = data_stream_buffer
        elif raw_data_point is not None:
            logger.info(f"Processing single raw data point: {raw_data_point}")
            if not isinstance(raw_data_point, list) or len(raw_data_point) != config.N_FEATURES:
                return {"status": "error", "message": f"raw_data_point must be a list of {config.N_FEATURES} sensor readings."}
            
            new_point_np = np.array(raw_data_point).reshape(1, -1)
            _sensor_reading_buffer = np.vstack((_sensor_reading_buffer, new_point_np))

            if _sensor_reading_buffer.shape[0] > config.SEQUENCE_LENGTH:
                _sensor_reading_buffer = _sensor_reading_buffer[-config.SEQUENCE_LENGTH:]
            
            if _sensor_reading_buffer.shape[0] < config.SEQUENCE_LENGTH:
                logger.info(f"Buffer filling: {_sensor_reading_buffer.shape[0]}/{config.SEQUENCE_LENGTH} points. No detection performed yet.")
                return {"status": "info", "message": f"Buffer filling: {_sensor_reading_buffer.shape[0]}/{config.SEQUENCE_LENGTH} points. Detection will start once buffer is full."}
            
            data_to_form_sequences = _sensor_reading_buffer

        else:
            logger.info(f"Processing data from test file: {config.TEST_DATA_PATH}")
            pass 


        # 3. Process data for detection
        if data_to_form_sequences is not None:
            X_detection_sequences, _ = load_and_preprocess_data(data_stream=data_to_form_sequences, scaler=scaler)
        else:
            X_detection_sequences, _ = load_and_preprocess_data(file_path=config.TEST_DATA_PATH, scaler=scaler)

        if X_detection_sequences.shape[0] == 0:
             return {"status": "error", "message": "No sequences could be formed from the provided data for detection."}


        X_detection_tensor = torch.from_numpy(X_detection_sequences).float().to(device)


        # 4. Perform detection
        logger.info("Performing inference for anomaly detection.")
        with torch.no_grad():
            if onnx_session:
                input_name = onnx_session.get_inputs()[0].name
                output_name = onnx_session.get_outputs()[0].name
                detection_input = X_detection_sequences.astype(np.float32)
                detection_reconstruction = onnx_session.run([output_name], {input_name: detection_input})[0]
                detection_reconstruction = torch.from_numpy(detection_reconstruction)
            else:
                detection_reconstruction = model(X_detection_tensor)

            detection_loss = torch.mean(torch.abs(detection_reconstruction.cpu() - X_detection_tensor.cpu()), dim=(1, 2)).numpy()

        anomalies = detection_loss > threshold
        anomaly_indices = np.where(anomalies)[0].tolist()
        
        logger.info(f"Detection complete. Found {np.sum(anomalies)} anomalies.")
        if np.sum(anomalies) > 0:
            logger.info(f"Anomaly indices: {anomaly_indices}")

        return {
            "status": "success",
            "message": "Anomaly detection complete.",
            "anomaly_threshold": float(threshold),
            "number_of_anomalies": int(np.sum(anomalies)),
            "anomaly_indices": anomaly_indices
        }

    except Exception as e:
        logger.error(f"An error occurred during detection: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"An error occurred during detection: {str(e)}"}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # For standalone run
    logger.info("Running detect.py as standalone script.")

    logger.info("Running detection on test file...")
    results = detect()
    logger.info(f"CLI Test File Detection Results: {results}")

    logger.info("\nSimulating real-time detection (single points):")
    _sensor_reading_buffer = np.zeros((0, config.N_FEATURES)) 
    
    normal_point = np.random.rand(config.N_FEATURES).tolist()
    anomaly_point = (np.array(normal_point) + np.array([config.SPIKE_MAGNITUDE] + [0]*(config.N_FEATURES-1))).tolist()

    simulated_stream = []
    for _ in range(config.SEQUENCE_LENGTH - 5):
        simulated_stream.append(normal_point)
    for _ in range(5):
        simulated_stream.append(normal_point)
    simulated_stream.append(anomaly_point)
    
    for i, point in enumerate(simulated_stream):
        logger.info(f"Processing point {i+1}: {point}")
        detection_result = detect(raw_data_point=point)
        if detection_result["status"] == "success" and detection_result["number_of_anomalies"] > 0:
            logger.info(f"  Anomaly Detected! Indices: {detection_result['anomaly_indices']}")
        elif detection_result["status"] == "info":
            logger.info(f"  {detection_result['message']}")
        else:
            logger.info(f"  No anomaly detected.")
