import torch
from torch import nn
import sys
import os
import logging # Import logging

logger = logging.getLogger(__name__) # Get logger instance

# Add the parent directory to sys.path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=config.EMBEDDING_DIM):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
    def forward(self, x):
        x, _ = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((-1, self.embedding_dim))

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=config.EMBEDDING_DIM, n_features=config.N_FEATURES):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)
    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=config.EMBEDDING_DIM):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def create_autoencoder():
    return RecurrentAutoencoder(config.SEQUENCE_LENGTH, config.N_FEATURES, config.EMBEDDING_DIM)

def export_to_onnx(model, onnx_path, sequence_length, n_features):
    """
    Exports the PyTorch model to ONNX format.
    """
    logger.info(f"Attempting to export model to ONNX at {onnx_path}")
    try:
        # Create a dummy input for ONNX export
        dummy_input = torch.randn(1, sequence_length, n_features, requires_grad=True)
        
        torch.onnx.export(model,                    # model being run
                          dummy_input,              # model input (or a tuple for multiple inputs)
                          onnx_path,                # where to save the model (can be a file or file-like object)
                          export_params=True,       # store the trained parameter weights inside the model file
                          opset_version=11,         # the ONNX version to export the model to
                          do_constant_folding=True, # whether to execute constant folding for optimization
                          input_names = ['input'],   # the names for the input of the graph
                          output_names = ['output'], # the names for the output of the graph
                          dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                        'output' : {0 : 'batch_size'}})
        logger.info(f"Model successfully exported to ONNX at {onnx_path}")
    except Exception as e:
        logger.error(f"Error exporting model to ONNX at {onnx_path}. Error: {e}", exc_info=True)
        raise RuntimeError(f"Failed to export model to ONNX: {e}")
