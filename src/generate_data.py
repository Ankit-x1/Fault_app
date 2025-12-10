import os
import numpy as np
import pandas as pd
from scipy.signal import chirp

# Add the parent directory to sys.path to import config
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def generate_data():
    np.random.seed(42) # for reproducibility

    total_normal_points = config.NORMAL_DATA_POINTS
    total_anomaly_points = config.ANOMALY_DATA_POINTS
    
    # Generate time series data
    time = np.linspace(0, 100, total_normal_points + total_anomaly_points)

    # Base signals for IMU features
    # Accel: Simulate some oscillating motion
    accel_x_base = 0.5 * np.sin(time / 5)
    accel_y_base = 0.3 * np.cos(time / 3)
    accel_z_base = 9.81 + 0.1 * np.sin(time / 10) # Gravity component

    # Gyro: Simulate some rotation
    gyro_x_base = 0.1 * np.sin(time / 2)
    gyro_y_base = 0.05 * np.cos(time / 4)
    gyro_z_base = 0.02 * np.sin(time / 6)

    # Combine into a single array
    normal_data_base = np.array([accel_x_base, accel_y_base, accel_z_base,
                                  gyro_x_base, gyro_y_base, gyro_z_base]).T

    # Add Gaussian noise
    normal_data = normal_data_base + np.random.normal(0, config.NOISE_LEVEL, normal_data_base.shape)

    # Create normal dataset
    normal_df = pd.DataFrame(data=normal_data[:total_normal_points], columns=config.SENSOR_FEATURES)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    normal_df.to_csv(config.NORMAL_DATA_PATH, index=False)

    # Create test dataset with anomalies
    test_data = normal_data.copy()
    
    # Introduce spike anomalies in accel_x and gyro_y
    spike_indices_1 = np.random.randint(0, len(time), size=total_anomaly_points // 4)
    spike_indices_2 = np.random.randint(0, len(time), size=total_anomaly_points // 4)
    test_data[spike_indices_1, 0] += config.SPIKE_MAGNITUDE * np.random.choice([-1, 1], size=len(spike_indices_1))
    test_data[spike_indices_2, 4] += config.SPIKE_MAGNITUDE * np.random.choice([-1, 1], size=len(spike_indices_2))

    # Introduce drift anomalies in accel_z and gyro_x
    drift_start_1 = np.random.randint(0, len(time) - 100)
    test_data[drift_start_1:drift_start_1+50, 2] += np.linspace(0, config.DRIFT_MAGNITUDE, 50)
    drift_start_2 = np.random.randint(0, len(time) - 100)
    test_data[drift_start_2:drift_start_2+50, 3] += np.linspace(0, -config.DRIFT_MAGNITUDE, 50)

    test_df = pd.DataFrame(data=test_data, columns=config.SENSOR_FEATURES)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)
    
    print(f"Generated normal_data.csv and test_data.csv in {config.DATA_DIR}.")

if __name__ == '__main__':
    generate_data()