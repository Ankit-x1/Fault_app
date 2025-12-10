# Deployment Guide: IMU Fault Detection on Embedded Systems

This guide provides comprehensive instructions for deploying the IMU Fault Detection model onto an embedded system, focusing on real-time data acquisition, preprocessing, ONNX model inference, and anomaly detection.

## 1. Prerequisites for Embedded Device

Before proceeding, ensure your embedded device meets the following requirements:

*   **Hardware:**
    *   **IMU Sensor (MPU-6050 or similar 6-axis IMU):** Connected and configured for data acquisition.
    *   **Microcontroller/SBC:** Capable of running Python (e.g., Raspberry Pi, ESP32 running MicroPython, or other Linux-based embedded systems). Sufficient RAM and processing power for `onnxruntime` or an equivalent inference engine.
    *   **Storage:** Enough space to store the ONNX model (`autoencoder.onnx`), scaler (`scaler.pkl`), and your application script.
*   **Software (Python-based devices):**
    *   **Python 3.x:** Installed on the device.
    *   **`pip`:** Python package installer.
    *   **Required Libraries:**
        *   `numpy`
        *   `scikit-learn` (for `MinMaxScaler` if recreating or loading it; `joblib` needs `scikit-learn` for loading the scaler)
        *   `joblib` (for loading the scaler)
        *   `onnxruntime` (for running ONNX models efficiently)
        *   `smbus` (for I2C communication with MPU-6050 on Linux/Raspberry Pi) or `machine` (for MicroPython on ESP32).
        *   `pyserial` (if communicating via serial with a different sensor setup).

## 2. Prepare Model and Scaler for Device

The training process (via `python main.py train` or `/api/train` endpoint) generates the necessary files:

*   `models/autoencoder.onnx`: The ONNX format of the trained autoencoder model. This is optimized for inference.
*   `models/scaler.pkl`: The `MinMaxScaler` object, serialized using `joblib`, used to preprocess sensor data to the same scale as during training.

**Transfer these two files (`autoencoder.onnx` and `scaler.pkl`) to your embedded device.** Place them in a location accessible by your embedded application, ideally in a `models/` subdirectory relative to your script.

## 3. Embedded Device Application (Conceptual Python Script)

The following Python script outlines the logic for real-time fault detection on an embedded device. This is a conceptual example; specific IMU interaction (`read_imu_data` function) will vary based on your device's operating system and libraries.

```python
import numpy as np
import joblib
import onnxruntime as rt
import time
import os
import collections

# --- Configuration (should match config.py on host) ---
# Adjust paths as per your device's file system
ONNX_MODEL_PATH = "models/autoencoder.onnx"
SCALER_PATH = "models/scaler.pkl"

SEQUENCE_LENGTH = 50  # Must match the sequence length used during training
N_FEATURES = 6        # Accel_x,y,z, Gyro_x,y,z
ANOMALY_THRESHOLD_MULTIPLIER = 2.5 # Must match config.py

# Placeholder for the anomaly threshold calculated from training data
# This value should be pre-calculated on the host (e.g., by running `python main.py detect` or API)
# and hardcoded or loaded from a configuration file on the embedded device.
# For demonstration, let's use a dummy value. In production, REPLACE THIS.
PRE_CALCULATED_ANOMALY_THRESHOLD = 0.0245 # Example: Replace with your actual calculated threshold

# --- Initialize ---
print("Initializing embedded fault detection system...")

# Load the scaler
try:
    scaler = joblib.load(SCALER_PATH)
    print(f"Scaler loaded from {SCALER_PATH}")
except Exception as e:
    print(f"ERROR: Could not load scaler: {e}")
    exit()

# Load the ONNX model
try:
    onnx_session = rt.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    print(f"ONNX model loaded from {ONNX_MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Could not load ONNX model: {e}")
    exit()

# Data buffer for forming sequences
# collections.deque is efficient for appending/popping from ends
sensor_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)

# --- IMU Data Acquisition (Conceptual) ---
def read_imu_data():
    """
    Conceptual function to read a single data point from the IMU 6050.
    In a real implementation, this would involve I2C communication.
    Returns: A list of 6 float values: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    """
    # Replace this with actual IMU reading code (e.g., using smbus, machine, etc.)
    # Example: Simulating IMU data
    accel_x = np.random.uniform(-1.0, 1.0)
    accel_y = np.random.uniform(-1.0, 1.0)
    accel_z = np.random.uniform(9.0, 10.0) # Gravity component around 9.81
    gyro_x = np.random.uniform(-0.1, 0.1)
    gyro_y = np.random.uniform(-0.1, 0.1)
    gyro_z = np.random.uniform(-0.1, 0.1)
    return [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]

# --- Main Loop for Real-time Detection ---
print("\nStarting real-time fault detection loop...")
print(f"Anomaly Threshold: {PRE_CALCULATED_ANOMALY_THRESHOLD:.4f}")

try:
    while True:
        # 1. Acquire data
        current_readings = read_imu_data()
        sensor_buffer.append(current_readings)

        # 2. Check if buffer is full
        if len(sensor_buffer) < SEQUENCE_LENGTH:
            print(f"Buffer filling: {len(sensor_buffer)}/{SEQUENCE_LENGTH} samples.")
            time.sleep(0.1) # Simulate sensor reading interval
            continue

        # 3. Preprocess data
        # Convert deque to numpy array
        sequence_np = np.array(list(sensor_buffer)).reshape(1, SEQUENCE_LENGTH, N_FEATURES)
        
        # Scale the sequence using the loaded scaler
        # The scaler expects 2D input (num_samples, num_features)
        scaled_sequence_flat = scaler.transform(sequence_np.reshape(-1, N_FEATURES))
        scaled_sequence = scaled_sequence_flat.reshape(1, SEQUENCE_LENGTH, N_FEATURES)

        # 4. ONNX Inference
        input_data = scaled_sequence.astype(np.float32)
        onnx_output = onnx_session.run([output_name], {input_name: input_data})[0]
        reconstruction_np = onnx_output # Keep as numpy for loss calc on device

        # 5. Calculate Reconstruction Error (MAE)
        reconstruction_loss = np.mean(np.abs(reconstruction_np - scaled_sequence), axis=(1, 2))

        # 6. Anomaly Detection
        is_anomaly = reconstruction_loss[0] > PRE_CALCULATED_ANOMALY_THRESHOLD

        if is_anomaly:
            print(f"[{time.strftime('%H:%M:%S')}] !!! ANOMALY DETECTED !!! Loss: {reconstruction_loss[0]:.4f}")
            # --- Trigger Alert/Action Here ---
            # e.g., turn on an LED, send a warning message, log to persistent storage
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Normal operation. Loss: {reconstruction_loss[0]:.4f}")

        time.sleep(0.1) # Simulate sensor reading interval (e.g., 100ms)

except KeyboardInterrupt:
    print("\nDetection stopped by user.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

```

## 4. How to Determine `PRE_CALCULATED_ANOMALY_THRESHOLD`

The `PRE_CALCULATED_ANOMALY_THRESHOLD` is a crucial parameter for on-device detection. It should be determined **offline** (on your development machine) using the `detect` functionality of this application:

1.  **Train the model:** Run `python main.py train` or trigger the `/api/train` endpoint. This will train the model and save the scaler.
2.  **Run detection on normal data:** Execute `python main.py detect`. The output will include the `Anomaly threshold` calculated from the normal training data. Copy this value.
3.  **Refine (Optional but Recommended):** You might want to fine-tune this threshold by testing it against known anomalous data in a controlled environment to balance false positives and false negatives.
4.  **Update `DEPLOYMENT_GUIDE.md`** with the actual threshold or load it from a separate `config_device.py` on the embedded system.

## 5. Next Steps for Production Deployment

*   **Robust IMU Driver:** Implement a robust driver for your specific IMU 6050 (or other sensor) on the chosen embedded platform.
*   **Persistent Storage:** For embedded systems, consider logging anomalies to persistent storage (e.g., SD card, flash memory) rather than just printing.
*   **Communication:** If the device needs to report anomalies, integrate wireless communication (Wi-Fi, Bluetooth, LoRaWAN) to a central monitoring system.
*   **Error Handling & Watchdogs:** Implement comprehensive error handling and watchdog timers to ensure the application's resilience.
*   **Power Management:** Optimize the application for low power consumption, crucial for battery-powered devices.
*   **Firmware Integration:** For very constrained devices, the ONNX model might need further conversion (e.g., to TensorFlow Lite Micro) and integration directly into device firmware (C/C++).
