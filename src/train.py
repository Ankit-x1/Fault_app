import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
import logging # Import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from data_loader import load_and_preprocess_data
from model import create_autoencoder, export_to_onnx

def train():
    logger.info("Starting model training...")
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load and preprocess the normal data
    logger.info(f"Loading and preprocessing data from {config.NORMAL_DATA_PATH}")
    X_train, scaler = load_and_preprocess_data(file_path=config.NORMAL_DATA_PATH)
    
    # Save the scaler
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, config.SCALER_PATH)
    logger.info(f"Scaler saved to {config.SCALER_PATH}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Create the autoencoder model
    model = create_autoencoder().to(device)
    logger.info("Autoencoder model created.")

    # Define loss function and optimizer
    criterion = nn.L1Loss(reduction='sum') # MAE Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Train the model
    training_log = []
    for epoch in range(config.NUM_EPOCHS):
        for data in train_loader:
            seq, _ = data
            seq = seq.to(device)
            # Forward pass
            output = model(seq)
            loss = criterion(output, seq)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        log_message = f'Epoch [{epoch+1}/{config.NUM_EPOCHS}], Loss: {loss.item():.4f}'
        logger.info(log_message)
        training_log.append(log_message)

    # Save the PyTorch model
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), config.PYTORCH_MODEL_PATH)
    logger.info(f"PyTorch Model saved to {config.PYTORCH_MODEL_PATH}")
    
    # Export to ONNX
    export_to_onnx(model, config.ONNX_MODEL_PATH, config.SEQUENCE_LENGTH, config.N_FEATURES)
    logger.info(f"ONNX Model saved to {config.ONNX_MODEL_PATH}")

    return "\n".join(training_log) + f"\nPyTorch Model saved to {config.PYTORCH_MODEL_PATH}\nONNX Model saved to {config.ONNX_MODEL_PATH}\nScaler saved to {config.SCALER_PATH}"