# Fault Detection using Autoencoders (PyTorch & Flask)

This project demonstrates a fault detection system for an embedded device using an autoencoder neural network. The application is built with Python, PyTorch, and Flask.

## Project Structure

```
APPAnkit/
├── data/
│   ├── normal_data.csv
│   └── test_data.csv
├── models/
│   ├── autoencoder.pth
│   └── scaler.pkl
├── src/
│   ├── data_loader.py
│   ├── detect.py
│   ├── generate_data.py
│   ├── model.py
│   └── train.py
├── templates/
│   └── index.html
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup

1.  **Clone the repository or download the project files.**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Generate the synthetic data:**
    Although the data is already provided, you can regenerate it by running:
    ```bash
    python src/generate_data.py
    ```

## Usage

### Command Line Interface (CLI)

The application can still be run via the command line for training or detection on file data:

*   **Train the Model:**
    ```bash
    python main.py train
    ```
    This will train the autoencoder model on `normal_data.csv` and save it as `models/autoencoder.pth` along with the scaler as `models/scaler.pkl`.

*   **Detect Faults (on test_data.csv):**
    ```bash
    python main.py detect
    ```
    This will load the trained model and scaler, and identify anomalies in `test_data.csv`. The script will output the detected anomalies to the console.

### Web Application (SaaS)

To run the web application and interact with it via a browser:

1.  **Start the Flask server:**
    ```bash
    python app.py
    ```
    The application will typically run on `http://127.0.0.1:5000/` (or `localhost:5000`).

2.  **Access the web interface:**
    Open your web browser and navigate to `http://127.0.0.1:5000/`.

    *   **Train Model:** Click the "Start Training" button on the web page. This will trigger the model training process via the `/api/train` endpoint.
    *   **Detect Faults:** Enter a comma-separated list of sensor readings into the provided text area and click "Detect Anomalies". This will send the data to the `/api/detect` endpoint and display the detected anomalies or a "no anomalies found" message.