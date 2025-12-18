import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import joblib
import logging
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class DetectionResult:
    is_anomaly: bool
    confidence: float
    reconstruction_error: float
    statistical_score: float
    ensemble_score: float
    sensor_importance: Dict[str, float]


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return torch.sum(x * attention_weights, dim=1), attention_weights


class IndustrialFaultDetector(nn.Module):
    def __init__(self, n_features: int, seq_len: int = 50, embedding_dim: int = 64):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim * 2,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )

        self.attention = AttentionLayer(embedding_dim * 4)

        self.decoder_lstm = nn.LSTM(
            input_size=embedding_dim * 4,
            hidden_size=embedding_dim * 2,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, n_features),
        )

        self.threshold_net = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        lstm_out, _ = self.encoder_lstm(x)
        encoded, attention_weights = self.attention(lstm_out)
        return encoded, attention_weights

    def decode(self, encoded):
        repeated = encoded.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded, _ = self.decoder_lstm(repeated)
        output = self.output_layer(decoded)
        return output

    def forward(self, x):
        encoded, attention_weights = self.encode(x)
        reconstructed = self.decode(encoded)
        adaptive_threshold = self.threshold_net(encoded)
        return reconstructed, attention_weights, adaptive_threshold


class StatisticalProcessControl:
    def __init__(self, window_size: int = 100, alpha: float = 0.001):
        self.window_size = window_size
        self.alpha = alpha
        self.data_buffer = deque(maxlen=window_size)
        self.pca = PCA(n_components=0.95)

    def update(self, data: np.ndarray):
        self.data_buffer.extend(data)
        if len(self.data_buffer) >= self.window_size:
            self.pca.fit(np.array(self.data_buffer))

    def calculate_control_limits(self) -> Tuple[float, float]:
        if len(self.data_buffer) < self.window_size:
            return -3.0, 3.0

        data_array = np.array(self.data_buffer)
        pca_scores = self.pca.transform(data_array)
        t2_scores = np.sum(pca_scores**2 / self.pca.explained_variance_, axis=1)

        n_samples, n_components = pca_scores.shape
        f_critical = 3.0
        ucl = ((n_samples - 1) * n_components / (n_samples - n_components)) * f_critical
        lcl = 0.0

        return lcl, ucl

    def detect_anomaly(self, data: np.ndarray) -> Tuple[bool, float]:
        if len(self.data_buffer) < self.window_size:
            return False, 0.0

        try:
            pca_score = self.pca.transform(data.reshape(1, -1))
            t2_score = np.sum(pca_score**2 / self.pca.explained_variance_, axis=1)[0]

            lcl, ucl = self.calculate_control_limits()
            is_anomaly = t2_score > ucl
            confidence = min(t2_score / ucl, 2.0) if ucl > 0 else 0.0

            return is_anomaly, confidence
        except:
            return False, 0.0


class FaultDetectionSystem:
    def __init__(self, n_features: int = 6, seq_len: int = 50):
        self.n_features = n_features
        self.seq_len = seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = IndustrialFaultDetector(n_features, seq_len).to(self.device)
        self.scaler = MinMaxScaler()

        self.spc = StatisticalProcessControl()
        self.isolation_forest = IsolationForest(contamination=0.05, random_state=42)

        self.reconstruction_threshold = 0.0
        self.ensemble_threshold = 0.65

        self.feature_names = [
            "accel_x",
            "accel_y",
            "accel_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
        ]

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def train(self, data: np.ndarray, epochs: int = 50, batch_size: int = 64) -> Dict:
        self.logger.info("Starting industrial-grade training...")

        data_scaled = self.scaler.fit_transform(data)

        X = []
        for i in range(len(data_scaled) - self.seq_len + 1):
            X.append(data_scaled[i : i + self.seq_len])
        X = np.array(X)

        self.isolation_forest.fit(data_scaled)

        X_tensor = torch.FloatTensor(X).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        criterion = nn.L1Loss()

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i : i + batch_size]

                reconstructed, attention_weights, adaptive_threshold = self.model(batch)
                reconstruction_loss = criterion(reconstructed, batch)

                threshold_loss = torch.mean(adaptive_threshold)
                total_loss = reconstruction_loss + 0.1 * threshold_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += reconstruction_loss.item()

            avg_loss = epoch_loss / (len(X_tensor) // batch_size)
            losses.append(avg_loss)
            scheduler.step(avg_loss)

            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

        self.model.eval()
        with torch.no_grad():
            reconstructed, _, _ = self.model(X_tensor)
            reconstruction_errors = torch.mean(
                torch.abs(reconstructed - X_tensor), dim=(1, 2)
            )
            self.reconstruction_threshold = torch.quantile(
                reconstruction_errors, 0.95
            ).item()

        self.logger.info(
            f"Training complete. Threshold: {self.reconstruction_threshold:.6f}"
        )
        return {"losses": losses, "threshold": self.reconstruction_threshold}

    def detect_realtime(self, sensor_data: List[float]) -> DetectionResult:
        if len(sensor_data) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {len(sensor_data)}"
            )

        data_normalized = self.scaler.transform([sensor_data])[0]

        self.spc.update([data_normalized])

        self.model.eval()
        with torch.no_grad():
            data_tensor = (
                torch.FloatTensor(data_normalized)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )

            sequence = data_tensor.repeat(1, self.seq_len, 1)

            reconstructed, attention_weights, adaptive_threshold = self.model(sequence)
            reconstruction_error = torch.mean(
                torch.abs(reconstructed - sequence)
            ).item()

            feature_importance = dict(
                zip(self.feature_names, attention_weights.squeeze().cpu().numpy())
            )

        is_stat_anomaly, stat_confidence = self.spc.detect_anomaly(data_normalized)

        nn_anomaly_score = min(
            reconstruction_error / self.reconstruction_threshold, 1.8
        )
        ensemble_score = nn_anomaly_score * 0.55 + stat_confidence * 0.45

        is_anomaly = ensemble_score > self.ensemble_threshold

        return DetectionResult(
            is_anomaly=is_anomaly,
            confidence=ensemble_score,
            reconstruction_error=reconstruction_error,
            statistical_score=stat_confidence,
            ensemble_score=ensemble_score,
            sensor_importance=feature_importance,
        )

    def save_model(self, path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "scaler": self.scaler,
                "reconstruction_threshold": self.reconstruction_threshold,
                "ensemble_threshold": self.ensemble_threshold,
                "spc_buffer": list(self.spc.data_buffer),
                "isolation_forest": self.isolation_forest,
            },
            path,
        )
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.scaler = checkpoint["scaler"]
        self.reconstruction_threshold = checkpoint["reconstruction_threshold"]
        self.ensemble_threshold = checkpoint["ensemble_threshold"]
        self.spc.data_buffer.extend(checkpoint["spc_buffer"])
        self.isolation_forest = checkpoint["isolation_forest"]
        self.logger.info(f"Model loaded from {path}")


def load_data_from_csv(file_path: str) -> np.ndarray:
    try:
        df = pd.read_csv(file_path)
        required_features = [
            "accel_x",
            "accel_y",
            "accel_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
        ]

        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        return df[required_features].values
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")


def main():
    detector = FaultDetectionSystem()

    try:
        train_data = load_data_from_csv("data/normal_data.csv")
        print(f"Loaded {len(train_data)} training samples")

        training_results = detector.train(train_data, epochs=30)
        print(f"Training complete. Final loss: {training_results['losses'][-1]:.6f}")

        detector.save_model("models/industrial_fault_detector.pth")

        print("\nSystem validation:")
        validation_samples = min(10, len(train_data))
        normal_count = 0
        for i in range(validation_samples):
            sample = train_data[i].tolist()
            result = detector.detect_realtime(sample)
            if not result.is_anomaly:
                normal_count += 1

        print(
            f"Normal samples correctly identified: {normal_count}/{validation_samples}"
        )
        print("Industrial Fault Detection System ready.")

    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()
