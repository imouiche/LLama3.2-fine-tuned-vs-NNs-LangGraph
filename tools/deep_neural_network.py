import numpy as np  # Numerical operations
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network modules
import torch.optim as optim  
from torch.utils.data import DataLoader, TensorDataset  
from torch.optim.lr_scheduler import CosineAnnealingLR  
from sklearn.feature_extraction.text import HashingVectorizer  
import logging  # For logging messages


class ResidualBlock(nn.Module):
    
    def __init__(self, hidden_size, dropout_prob):

        super(ResidualBlock, self).__init__()
        
        # Main transformation path (2 linear layers with normalization)
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # First linear transformation
            nn.LayerNorm(hidden_size),  # Normalize across features (stabilizes training)
            nn.ReLU(),  # Non-linear activation (Rectified Linear Unit)
            nn.Dropout(dropout_prob),  # Randomly drop units to prevent overfitting
            nn.Linear(hidden_size, hidden_size),  # Second linear transformation
            nn.LayerNorm(hidden_size),  # Second normalization
        )
        
        # Final activation after adding skip connection
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x  # Save input for skip connection
        out = self.block(x)  # Pass through the transformation
        out += residual  # Add skip connection (element-wise addition)
        return self.relu(out)  # Final activation


class DeepNeuralNetwork(nn.Module):
    
    def __init__(self, input_size, num_layers=10, hidden_size=4096, dropout_prob=0.2):
        super(DeepNeuralNetwork, self).__init__()

        # First layer: Project input features to hidden dimension
        # input_size (5000) → hidden_size (4096)
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Linear projection
            nn.LayerNorm(hidden_size),  # Normalize for stable training
            nn.ReLU(),  # Non-linear activation
            nn.Dropout(dropout_prob),  # Regularization
        )

        # Stack of residual blocks (num_layers - 2 blocks)
        # Why num_layers - 2? Because we have input layer + output layer
        # Default: 10 - 2 = 8 residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers - 2):
            self.residual_blocks.append(ResidualBlock(hidden_size, dropout_prob))

        # Output layer: Project hidden dimension to single value (price)
        # hidden_size (4096) → 1 (price)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):

        # Project input features to hidden dimension
        x = self.input_layer(x)

        # Pass through each residual block
        for block in self.residual_blocks:
            x = block(x)

        # Project to single output value (price)
        return self.output_layer(x)

Y_STD = 1.0328539609909058  # Standard deviation of log(price + 1) in training data
Y_MEAN = 4.434937953948975  # Mean of log(price + 1) in training data


class DeepNeuralNetworkInference:
    
    def __init__(self):

        self.vectorizer = None  # HashingVectorizer for text → features
        self.model = None  # The neural network model
        self.device = None  # Compute device (CPU/GPU)

        # Set random seeds for reproducibility
        # Ensures same results across runs with same input
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def setup(self):
        self.vectorizer = HashingVectorizer(
            n_features=5000,  # Output vector size
            stop_words="english",  # Remove common words
            binary=True  # Binary features (presence/absence)
        )
        self.model = DeepNeuralNetwork(5000)
        
        # Determine best available device for computation
        if torch.cuda.is_available():
            # NVIDIA GPU available (fastest)
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            # Apple Silicon GPU available (M1/M2/M3 Macs)
            self.device = torch.device("mps")
        else:
            # Fall back to CPU
            self.device = torch.device("cpu")

        logging.info(f"Neural Network is using {self.device}")

        # Move model to the selected device
        # Use to_empty() to avoid meta tensor issues on first load
        try:
            self.model.to(self.device)
        except RuntimeError as e:
            if "meta tensor" in str(e):
                # Fallback for meta tensor issue: use to_empty then move
                self.model = self.model.to_empty(device=self.device)
            else:
                raise

    def load(self, path):
        # Load model weights from file
        self.model.load_state_dict(torch.load(path))
        
        # Move model to device (in case device changed)
        self.model.to(self.device)

    def inference(self, text: str) -> float:
        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()
        
        # Disable gradient computation (saves memory, speeds up inference)
        with torch.no_grad():
            # transform() returns sparse matrix of shape (1, 5000)
            vector = self.vectorizer.transform([text])
            # .to(device) moves tensor to GPU/CPU
            vector = torch.FloatTensor(vector.toarray()).to(self.device)

            pred = self.model(vector)[0]  # Extract scalar value
    
            result = torch.exp(pred * Y_STD + Y_MEAN) - 1
            
            # Convert PyTorch tensor to Python float
            result = result.item()
        
        # Step 5: Ensure non-negative price (clamp to 0 if negative)
        return max(0, result)