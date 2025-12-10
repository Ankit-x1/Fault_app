import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
NORMAL_DATA_PATH = os.path.join(DATA_DIR, 'normal_data.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data.csv')

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PYTORCH_MODEL_PATH = os.path.join(MODELS_DIR, 'autoencoder.pth')
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, 'autoencoder.onnx')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

# Sensor parameters (IMU 6050)
SENSOR_FEATURES = [
    'accel_x', 'accel_y', 'accel_z',
    'gyro_x', 'gyro_y', 'gyro_z'
]
N_FEATURES = len(SENSOR_FEATURES) # Number of features from the IMU sensor

# Model hyperparameters
SEQUENCE_LENGTH = 50 # Increased for more context in time series
EMBEDDING_DIM = 64
NUM_EPOCHS = 20 # Decreased for faster training
BATCH_SIZE = 64 # Increased for faster training
LEARNING_RATE = 1e-3

# Anomaly detection parameters
ANOMALY_THRESHOLD_MULTIPLIER = 2.5 # Multiplier for standard deviation to set threshold

# Data generation parameters (for IMU 6050 simulation)
# These will be used in generate_data.py
NORMAL_DATA_POINTS = 5000
ANOMALY_DATA_POINTS = 500
NOISE_LEVEL = 0.02 # General noise in sensor readings
SPIKE_MAGNITUDE = 1.5 # Magnitude of anomaly spikes
DRIFT_MAGNITUDE = 0.5 # Magnitude of anomaly drift

# Note: For production deployments, consider loading sensitive or environment-specific
# configurations from environment variables (e.g., using os.getenv) or external
# configuration files (e.g., YAML, JSON) to enhance flexibility and security.

