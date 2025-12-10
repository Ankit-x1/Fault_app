import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Add the parent directory to sys.path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_and_preprocess_data(file_path=None, data_stream=None, scaler=None):
    """
    Loads data from a CSV file or processes a numpy array data stream.
    Scales the data and creates sequences.

    Args:
        file_path (str, optional): Path to the CSV file. Defaults to None.
        data_stream (np.ndarray, optional): A 2D numpy array (num_samples, num_features) of sensor readings.
                                            Used for processing real-time-like data. Defaults to None.
        scaler (MinMaxScaler, optional): An already fitted MinMaxScaler instance. If None, a new one is fitted.
                                         Defaults to None.

    Returns:
        tuple: A tuple containing:
               - np.ndarray: The 3D array of sequences (num_sequences, SEQUENCE_LENGTH, N_FEATURES).
               - MinMaxScaler: The fitted or provided scaler instance.
    """
    if file_path:
        df = pd.read_csv(file_path)
        data = df[config.SENSOR_FEATURES].values
    elif data_stream is not None:
        if not isinstance(data_stream, np.ndarray) or data_stream.ndim != 2:
            raise ValueError("data_stream must be a 2D numpy array (num_samples, num_features).")
        if data_stream.shape[1] != config.N_FEATURES:
            raise ValueError(f"data_stream has {data_stream.shape[1]} features, expected {config.N_FEATURES}.")
        data = data_stream
    else:
        raise ValueError("Either 'file_path' or 'data_stream' must be provided.")

    # Scale the data
    if scaler is None:
        new_scaler = MinMaxScaler()
        data_scaled = new_scaler.fit_transform(data)
        current_scaler = new_scaler
    else:
        data_scaled = scaler.transform(data)
        current_scaler = scaler

    # Create sequences
    X = []
    for i in range(len(data_scaled) - config.SEQUENCE_LENGTH + 1):
        X.append(data_scaled[i : i + config.SEQUENCE_LENGTH])

    if not X:
        # This can happen if data_stream has fewer samples than SEQUENCE_LENGTH
        raise ValueError(f"Not enough data points ({len(data_scaled)}) to form sequences with SEQUENCE_LENGTH={config.SEQUENCE_LENGTH}.")
    
    return np.array(X), current_scaler

