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
    training_log = []
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load and preprocess the normal data
        logger.info(f"Loading and preprocessing data from {config.NORMAL_DATA_PATH}")
        try:
            X_train, scaler = load_and_preprocess_data(file_path=config.NORMAL_DATA_PATH)
        except DataLoaderError as e:
            logger.error(f"Error loading or preprocessing data: {e}", exc_info=True)
            raise RuntimeError(f"Error loading or preprocessing data: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
            raise RuntimeError(f"An unexpected error occurred during data loading: {e}")
        
        # Save the scaler
        try:
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            joblib.dump(scaler, config.SCALER_PATH)
            logger.info(f"Scaler saved to {config.SCALER_PATH}")
            training_log.append(f"Scaler saved to {config.SCALER_PATH}")
        except Exception as e:
            logger.error(f"Error saving scaler to {config.SCALER_PATH}. Error: {e}", exc_info=True)
            raise RuntimeError(f"Error saving scaler: {e}")
        
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
        logger.info(f"Starting model training for {config.NUM_EPOCHS} epochs.")
        try:
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
        except RuntimeError as e: # Catch specific PyTorch runtime errors, e.g., CUDA out of memory
            logger.error(f"PyTorch Runtime Error during training: {e}", exc_info=True)
            raise RuntimeError(f"Training failed due to runtime error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during model training: {e}", exc_info=True)
            raise RuntimeError(f"An unexpected error occurred during model training: {e}")

        # Save the PyTorch model
        try:
            os.makedirs(config.MODELS_DIR, exist_ok=True)
            torch.save(model.state_dict(), config.PYTORCH_MODEL_PATH)
            logger.info(f"PyTorch Model saved to {config.PYTORCH_MODEL_PATH}")
            training_log.append(f"PyTorch Model saved to {config.PYTORCH_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error saving PyTorch model to {config.PYTORCH_MODEL_PATH}. Error: {e}", exc_info=True)
            raise RuntimeError(f"Error saving PyTorch model: {e}")
        
        # Export to ONNX
        try:
            export_to_onnx(model, config.ONNX_MODEL_PATH, config.SEQUENCE_LENGTH, config.N_FEATURES)
            logger.info(f"ONNX Model saved to {config.ONNX_MODEL_PATH}")
            training_log.append(f"ONNX Model saved to {config.ONNX_MODEL_PATH}")
        except RuntimeError as e:
            logger.error(f"Failed to export model to ONNX: {e}", exc_info=True)
            training_log.append(f"Warning: ONNX Model export failed: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during ONNX export: {e}", exc_info=True)
            training_log.append(f"Warning: ONNX Model export failed due to an unexpected error: {e}")

        return "\n".join(training_log)

    except RuntimeError as e:
        logger.error(f"Training process failed: {e}", exc_info=True)
        raise # Re-raise the RuntimeError for the API to catch
    except Exception as e:
        logger.error(f"An unhandled error occurred in the train function: {e}", exc_info=True)
        raise RuntimeError(f"An unhandled error occurred during training: {e}")