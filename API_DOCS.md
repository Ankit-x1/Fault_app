# API Documentation for IMU Fault Detection Application

This document provides details for the RESTful API endpoints exposed by the Flask application, enabling interaction with the IMU Fault Detection system for model training and anomaly detection.

## Base URL

The API is served by the Flask application, typically running on `http://127.0.0.1:5000/` or a configured host.

---

## 1. Train Model

Initiates the training process for the autoencoder model using predefined normal sensor data.

*   **Endpoint:** `/api/train`
*   **Method:** `POST`
*   **Description:** Triggers the training of the PyTorch autoencoder model. The model is trained on `normal_data.csv` (simulated IMU data). Upon successful training, the trained PyTorch model (`autoencoder.pth`), its ONNX exported version (`autoencoder.onnx`), and the `MinMaxScaler` (`scaler.pkl`) are saved to the `models/` directory.
*   **Request:**
    *   **Headers:** `Content-Type: application/json` (though no body is strictly required for this endpoint).
    *   **Body:** (Optional) An empty JSON object `{}` is sufficient.
*   **Response:**
    *   **Success (200 OK):**
        ```json
        {
          "status": "success",
          "message": "Model training completed.",
          "log": "Detailed training log including epoch losses and model save paths."
        }
        ```
    *   **Error (500 Internal Server Error):**
        ```json
        {
          "status": "error",
          "message": "Error description."
        }
        ```
*   **Example (using `curl`):**
    ```bash
    curl -X POST http://127.0.0.1:5000/api/train -H "Content-Type: application/json" -d "{}"
    ```

---

## 2. Detect Faults

Performs real-time anomaly detection on incoming IMU sensor readings.

*   **Endpoint:** `/api/detect`
*   **Method:** `POST`
*   **Description:** Receives a stream of IMU sensor readings, processes them, and identifies anomalies. The system maintains an internal buffer of sensor readings to form sequences required by the autoencoder model (current sequence length: 50). Detection occurs only once enough data points are buffered.
*   **Request:**
    *   **Headers:** `Content-Type: application/json`
    *   **Body:** A JSON object containing `sensor_readings`.
        *   `sensor_readings` (array of numbers or array of arrays of numbers):
            *   **Single Data Point:** `[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]` (e.g., `[0.1, 0.2, 9.8, 0.01, 0.02, 0.005]`). This point is added to the internal buffer.
            *   **Multiple Data Points (Batch of Samples):** `[[ax1, ay1, az1, gx1, gy1, gz1], [ax2, ay2, az2, gx2, gy2, gz2], ...]` This array should represent a continuous stream of sensor readings from which sequences can be formed. If the number of samples in this array is less than `50`, an error will be returned.
*   **Response:**
    *   **Success (200 OK - Anomaly Detected/No Anomaly/Buffering):**
        ```json
        // If anomalies are detected
        {
          "status": "success",
          "message": "Anomaly detection complete.",
          "results": {
            "anomaly_threshold": 0.01234,
            "number_of_anomalies": 2,
            "anomaly_indices": [5, 12] // Indices are relative to the input sequence provided or buffer.
          }
        }
        // If no anomalies are detected
        {
          "status": "success",
          "message": "Anomaly detection complete.",
          "results": {
            "anomaly_threshold": 0.01234,
            "number_of_anomalies": 0,
            "anomaly_indices": []
          }
        }
        // If buffer is still filling (only for single data point input)
        {
          "status": "info",
          "message": "Buffer filling: 25/50 points. Detection will start once buffer is full.",
          "results": {} // results object will be empty or minimal
        }
        ```
    *   **Error (400 Bad Request / 500 Internal Server Error):**
        ```json
        {
          "status": "error",
          "message": "Error description (e.g., missing 'sensor_readings', invalid format, insufficient data)."
        }
        ```
*   **Example (using `curl` for a single data point):**
    ```bash
    curl -X POST http://127.0.0.1:5000/api/detect -H "Content-Type: application/json" -d '{"sensor_readings": [0.1, 0.2, 9.8, 0.01, 0.02, 0.005]}'
    ```
*   **Example (using `curl` for a batch of data points):**
    ```bash
    curl -X POST http://127.0.0.1:5000/api/detect -H "Content-Type: application/json" -d '{"sensor_readings": [[0.1,0.2,9.8,0.01,0.02,0.005], [0.11,0.21,9.8,0.01,0.02,0.005], ..., [0.1,0.2,9.8,0.01,0.02,0.005]]}'
    ```
    (Note: The `sensor_readings` array needs to contain at least `50` samples for detection to occur with `data_stream_buffer`.)

---
