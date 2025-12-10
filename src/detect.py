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



def _get_model_and_session(device):
    logger.info("Attempting to load model (preferring ONNX).")
    model = None
    session = None
    
    # Try loading ONNX model first
    if os.path.exists(config.ONNX_MODEL_PATH):
        try:
            session = rt.InferenceSession(config.ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
            logger.info(f"Loaded ONNX model from {config.ONNX_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Could not load ONNX model from {config.ONNX_MODEL_PATH}. Falling back to PyTorch. Error: {e}", exc_info=True)
            session = None # Reset session if ONNX fails

    # Fallback to PyTorch model if ONNX failed or not present
    if session is None: # Only attempt PyTorch if ONNX was not loaded
        if os.path.exists(config.PYTORCH_MODEL_PATH):
            try:
                model = create_autoencoder().to(device)
                model.load_state_dict(torch.load(config.PYTORCH_MODEL_PATH, map_location=device))
                model.eval()
                logger.info(f"Loaded PyTorch model from {config.PYTORCH_MODEL_PATH}")
            except Exception as e:
                logger.error(f"Error loading PyTorch model from {config.PYTORCH_MODEL_PATH}. Error: {e}", exc_info=True)
                raise RuntimeError(f"Failed to load PyTorch model: {e}")
        else:
            raise RuntimeError(f"No ONNX or PyTorch model found at {config.ONNX_MODEL_PATH} or {config.PYTORCH_MODEL_PATH}")
    
    return model, session


def detect(data_stream_buffer=None):
    logger.info("Starting anomaly detection.")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load the scaler
        try:
            scaler = joblib.load(config.SCALER_PATH)
            logger.info(f"Scaler loaded from {config.SCALER_PATH}")
        except FileNotFoundError as e:
            logger.error(f"Scaler file not found: {config.SCALER_PATH}. Error: {e}", exc_info=True)
            return {"status": "error", "message": f"Scaler file not found: {e}"}
        except Exception as e:
            logger.error(f"Error loading scaler from {config.SCALER_PATH}. Error: {e}", exc_info=True)
            return {"status": "error", "message": f"Error loading scaler: {e}"}

        # Load trained PyTorch model or ONNX session
        try:
            model, onnx_session = _get_model_and_session(device)
        except RuntimeError as e:
            logger.error(f"Model loading failed: {e}", exc_info=True)
            return {"status": "error", "message": f"Model loading failed: {e}"}
        except Exception as e:
            logger.error(f"An unexpected error occurred during model loading: {e}", exc_info=True)
            return {"status": "error", "message": f"An unexpected error occurred during model loading: {e}"}

        if model is None and onnx_session is None:
            return {"status": "error", "message": "No valid model (PyTorch or ONNX) could be loaded."}

        # 1. Determine threshold from normal training data
        logger.info("Determining anomaly threshold from normal data.")
        try:
            X_train_for_threshold, _ = load_and_preprocess_data(file_path=config.NORMAL_DATA_PATH, scaler=scaler)
            if X_train_for_threshold.shape[0] < config.SEQUENCE_LENGTH:
                return {"status": "error", "message": f"Normal data for threshold determination must contain at least {config.SEQUENCE_LENGTH} samples."}
        except DataLoaderError as e:
            logger.error(f"Error loading or preprocessing normal data for threshold: {e}", exc_info=True)
            return {"status": "error", "message": f"Error loading or preprocessing normal data for threshold: {e}"}
        except Exception as e:
            logger.error(f"An unexpected error occurred during threshold data processing: {e}", exc_info=True)
            return {"status": "error", "message": f"An unexpected error occurred during threshold data processing: {e}"}

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
            if data_stream_buffer.shape[0] != config.SEQUENCE_LENGTH:
                return {"status": "error", "message": f"Data stream buffer must contain exactly {config.SEQUENCE_LENGTH} samples for detection."}
            data_to_form_sequences = data_stream_buffer
        else:
            logger.info(f"Processing data from test file: {config.TEST_DATA_PATH}")
            pass # Use default test data if no stream is provided (for CLI/testing purposes) 


        # 3. Process data for detection
        try:
            if data_to_form_sequences is not None:
                X_detection_sequences, _ = load_and_preprocess_data(data_stream=data_to_form_sequences, scaler=scaler)
            else:
                X_detection_sequences, _ = load_and_preprocess_data(file_path=config.TEST_DATA_PATH, scaler=scaler)
            
            if X_detection_sequences.shape[0] == 0:
                return {"status": "error", "message": "No sequences could be formed from the provided data for detection."}
        except DataLoaderError as e:
            logger.error(f"Error loading or preprocessing detection data: {e}", exc_info=True)
            return {"status": "error", "message": f"Error loading or preprocessing detection data: {e}"}
        except Exception as e:
            logger.error(f"An unexpected error occurred during detection data processing: {e}", exc_info=True)
            return {"status": "error", "message": f"An unexpected error occurred during detection data processing: {e}"}


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

    logger.info("\nSimulating real-time detection (full buffer):")
    # To test real-time simulation, manually create a buffer of SEQUENCE_LENGTH
    # and pass it as data_stream_buffer
    sample_data_point = np.random.rand(config.N_FEATURES).tolist() # Example single point
    simulated_buffer = [sample_data_point] * config.SEQUENCE_LENGTH # Create a buffer

    logger.info(f"Processing a simulated buffer of {config.SEQUENCE_LENGTH} points.")
    detection_result = detect(data_stream_buffer=np.array(simulated_buffer))
    logger.info(f"Simulated Buffer Detection Results: {detection_result}")
