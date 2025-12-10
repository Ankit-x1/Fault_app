import os
import sys
from flask import Flask, render_template, request, jsonify
import numpy as np
import logging # Import logging

# Configure Flask app logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import config
import config

# Import existing functions
from train import train as train_model
from detect import detect as detect_faults

@app.route('/')
def index():
    logger.info("Serving index.html")
    return render_template(
        'index.html', 
        SEQUENCE_LENGTH=config.SEQUENCE_LENGTH, 
        N_FEATURES=config.N_FEATURES,
        NUM_EPOCHS=config.NUM_EPOCHS,
        SENSOR_FEATURES_STRING=", ".join(config.SENSOR_FEATURES)
    )

@app.route('/api/train', methods=['POST'])
def train_api():
    logger.info("API call received: /api/train")
    try:
        log_message = train_model()
        logger.info("Model training completed successfully.")
        return jsonify({"status": "success", "message": "Model training completed.", "log": log_message}), 200
    except Exception as e:
        logger.error(f"Error during training API call: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/detect', methods=['POST'])
def detect_api():
    logger.info("API call received: /api/detect")
    try:
        data = request.get_json()
        if 'sensor_readings' not in data:
            logger.error("Missing 'sensor_readings' in request body.")
            return jsonify({"status": "error", "message": "Missing 'sensor_readings' in request body. Expected a list of sensor data points."}), 400
        
        sensor_readings = data['sensor_readings']
        if not isinstance(sensor_readings, list):
            logger.error("'sensor_readings' must be a list.")
            return jsonify({"status": "error", "message": "'sensor_readings' must be a list of sensor data points or a single sensor data point (list of numbers)."}), 400

        sensor_readings_np = np.array(sensor_readings)

        if sensor_readings_np.ndim == 1:
            results = detect_faults(raw_data_point=sensor_readings_np.tolist())
        elif sensor_readings_np.ndim == 2:
            results = detect_faults(data_stream_buffer=sensor_readings_np)
        else:
            logger.error("Invalid format for 'sensor_readings'.")
            return jsonify({"status": "error", "message": "Invalid format for 'sensor_readings'. Expected a 1D or 2D array-like structure."}), 400
        
        logger.info(f"Detection API call results: {results['status']}")
        return jsonify({"status": results["status"], "message": results.get("message", "No message provided."), "results": {
            "anomaly_threshold": results.get("anomaly_threshold"),
            "number_of_anomalies": results.get("number_of_anomalies"),
            "anomaly_indices": results.get("anomaly_indices")
        }}), 200
    except Exception as e:
        logger.error(f"Error during detection API call: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Ensure the models directory exists
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    logger.info("Starting Flask application.")
    app.run(debug=True, host='0.0.0.0')