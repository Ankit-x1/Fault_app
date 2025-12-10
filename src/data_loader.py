import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import logging

logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class DataLoaderError(Exception):
    """Custom exception for errors during data loading or preprocessing."""
    pass

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
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                raise DataLoaderError(f"CSV file is empty: {file_path}")
            # Check if all required sensor features are in the DataFrame
            missing_features = [f for f in config.SENSOR_FEATURES if f not in df.columns]
            if missing_features:
                raise DataLoaderError(f"Missing sensor features in CSV file '{file_path}': {', '.join(missing_features)}")
            data = df[config.SENSOR_FEATURES].values
        except FileNotFoundError:
            logger.error(f"Data file not found: {file_path}", exc_info=True)
            raise DataLoaderError(f"Data file not found: {file_path}")
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty: {file_path}", exc_info=True)
            raise DataLoaderError(f"CSV file is empty: {file_path}")
        except KeyError as e:
            logger.error(f"Missing expected columns in CSV file '{file_path}': {e}", exc_info=True)
            raise DataLoaderError(f"Missing expected columns in CSV file '{file_path}': {e}")
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}", exc_info=True)
            raise DataLoaderError(f"Error reading CSV file {file_path}: {e}")
    elif data_stream is not None:
        if not isinstance(data_stream, np.ndarray) or data_stream.ndim != 2:
            raise DataLoaderError("data_stream must be a 2D numpy array (num_samples, num_features).")
        if data_stream.shape[1] != config.N_FEATURES:
            raise DataLoaderError(f"data_stream has {data_stream.shape[1]} features, expected {config.N_FEATURES}.")
        data = data_stream
    else:
        raise DataLoaderError("Either 'file_path' or 'data_stream' must be provided.")

    # Scale the data
    if scaler is None:
        new_scaler = MinMaxScaler()
        data_scaled = new_scaler.fit_transform(data)
        current_scaler = new_scaler
    else:
        try:
            data_scaled = scaler.transform(data)
        except Exception as e:
            logger.error(f"Error transforming data with provided scaler: {e}", exc_info=True)
            raise DataLoaderError(f"Error transforming data with provided scaler: {e}. Ensure scaler is fitted and data has correct dimensions.")
        current_scaler = scaler

    # Create sequences
    X = []
    for i in range(len(data_scaled) - config.SEQUENCE_LENGTH + 1):
        X.append(data_scaled[i : i + config.SEQUENCE_LENGTH])

    if not X:
        raise DataLoaderError(f"Not enough data points ({len(data_scaled)}) to form sequences with SEQUENCE_LENGTH={config.SEQUENCE_LENGTH}.")
    
    return np.array(X), current_scaler


