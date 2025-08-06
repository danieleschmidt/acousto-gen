"""
Neural hologram generation using generative models.
Implements VAE and GAN architectures for rapid acoustic hologram synthesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import time
import json

from .acoustic_field import AcousticField


@dataclass
class TrainingConfig:
    """Configuration for neural model training."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 100
    latent_dim: int = 128
    condition_dim: int = 32
    beta: float = 1.0  # β-VAE parameter
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class HologramDataset(Dataset):
    """Dataset for hologram training data."""
    
    def __init__(self, field_patterns: List[np.ndarray], conditions: List[Dict[str, Any]]):
        """
        Initialize dataset with field patterns and their conditions.
        
        Args:
            field_patterns: List of complex acoustic field arrays
            conditions: List of condition dictionaries (focal points, pressures, etc.)
        """
        self.patterns = []
        self.conditions = []
        
        for pattern, condition in zip(field_patterns, conditions):
            # Normalize and convert to real representation
            pattern_norm = self._normalize_pattern(pattern)
            condition_vector = self._vectorize_condition(condition)
            
            self.patterns.append(pattern_norm)
            self.conditions.append(condition_vector)
    
    def _normalize_pattern(self, pattern: np.ndarray) -> torch.Tensor:
        """Convert complex pattern to normalized real representation."""
        # Split into amplitude and phase
        amplitude = np.abs(pattern)
        phase = np.angle(pattern)
        
        # Normalize amplitude
        if np.max(amplitude) > 0:
            amplitude = amplitude / np.max(amplitude)
        
        # Normalize phase to [-1, 1]
        phase = phase / np.pi
        
        # Stack amplitude and phase channels
        pattern_real = np.stack([amplitude, phase], axis=0)
        
        return torch.tensor(pattern_real, dtype=torch.float32)
    
    def _vectorize_condition(self, condition: Dict[str, Any]) -> torch.Tensor:
        """Convert condition dictionary to vector representation."""
        vector = []
        
        # Focal points
        focal_points = condition.get('focal_points', [])
        for i in range(4):  # Support up to 4 focal points
            if i < len(focal_points):
                fp = focal_points[i]
                vector.extend(fp.get('position', [0, 0, 0]))
                vector.append(fp.get('pressure', 0) / 10000)  # Normalize pressure
            else:
                vector.extend([0, 0, 0, 0])  # Padding
        
        # Global parameters
        vector.append(condition.get('frequency', 40000) / 100000)  # Normalize frequency
        vector.append(condition.get('total_power', 1.0))
        vector.append(condition.get('focus_quality', 1.0))
        
        # Pad to fixed size
        while len(vector) < 32:
            vector.append(0.0)
        
        return torch.tensor(vector[:32], dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.patterns)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.patterns[idx], self.conditions[idx]


class HologramVAE(nn.Module):
    """
    Variational Autoencoder for hologram generation.
    
    Encodes acoustic field patterns into a latent space and decodes
    them conditioned on desired focal point specifications.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, ...] = (2, 32, 32, 32),  # [channels, x, y, z]
        latent_dim: int = 128,
        condition_dim: int = 32
    ):
        """
        Initialize VAE architecture.
        
        Args:
            input_shape: Shape of input field patterns
            latent_dim: Dimensionality of latent space
            condition_dim: Dimensionality of condition vector
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            # 3D convolutions for spatial features
            nn.Conv3d(2, 32, kernel_size=4, stride=2, padding=1),  # 16x16x16
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),  # 8x8x8
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),  # 4x4x4
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(128, 256, kernel_size=4, stride=1, padding=0),  # 1x1x1
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        )
        
        # Variational layers
        self.fc_mu = nn.Linear(256 + condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(256 + condition_dim, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim + condition_dim, 256 * 4 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8x8
            nn.BatchNorm3d(128),
            nn.ReLU(),
            
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16x16
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32x32
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            nn.ConvTranspose3d(32, 2, kernel_size=3, stride=1, padding=1),  # 32x32x32
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def encode(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        features = self.encoder(x)
        
        # Concatenate with condition
        combined = torch.cat([features, condition], dim=1)
        
        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Decode latent code to hologram."""
        # Combine latent code with condition
        combined = torch.cat([z, condition], dim=1)
        
        # Map to decoder input shape
        hidden = self.decoder_input(combined)
        hidden = hidden.view(-1, 256, 4, 4, 4)
        
        # Decode to hologram
        output = self.decoder(hidden)
        
        return output
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass through VAE."""
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, condition)
        
        return recon, mu, logvar
    
    def generate(self, condition: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Generate new holograms from condition."""
        self.eval()
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.latent_dim, device=condition.device)
            
            # Expand condition to match batch size
            if condition.dim() == 1:
                condition = condition.unsqueeze(0).expand(num_samples, -1)
            
            # Generate
            generated = self.decode(z, condition)
            
        return generated
    
    def interpolate(
        self,
        condition1: torch.Tensor,
        condition2: torch.Tensor,
        steps: int = 10
    ) -> torch.Tensor:
        """Interpolate between two conditions."""
        self.eval()
        with torch.no_grad():
            # Create interpolation weights
            alphas = torch.linspace(0, 1, steps, device=condition1.device)
            
            results = []
            for alpha in alphas:
                # Interpolate conditions
                interp_condition = (1 - alpha) * condition1 + alpha * condition2
                
                # Generate hologram
                generated = self.generate(interp_condition, num_samples=1)
                results.append(generated)
            
            return torch.cat(results, dim=0)


class HologramGAN(nn.Module):
    """
    Generative Adversarial Network for hologram synthesis.
    
    Uses conditional generation to create holograms matching
    specified acoustic field requirements.
    """
    
    def __init__(
        self,
        noise_dim: int = 128,
        condition_dim: int = 32,
        output_shape: Tuple[int, ...] = (2, 32, 32, 32)
    ):
        """
        Initialize GAN components.
        
        Args:
            noise_dim: Dimensionality of input noise
            condition_dim: Dimensionality of condition vector
            output_shape: Shape of generated holograms
        """
        super().__init__()
        
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.output_shape = output_shape
        
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
    
    def _build_generator(self) -> nn.Module:
        """Build generator network."""
        return nn.Sequential(
            # Input: noise + condition
            nn.Linear(self.noise_dim + self.condition_dim, 256 * 4 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4 * 4),
            nn.ReLU(),
            
            # Reshape to 3D
            nn.Unflatten(1, (256, 4, 4, 4)),
            
            # Upsampling layers
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8x8
            nn.BatchNorm3d(128),
            nn.ReLU(),
            
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16x16
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32x32
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            nn.ConvTranspose3d(32, 2, kernel_size=3, stride=1, padding=1),  # 32x32x32
            nn.Tanh()
        )
    
    def _build_discriminator(self) -> nn.Module:
        """Build discriminator network."""
        return nn.Sequential(
            # Input: hologram + condition (embedded)
            nn.Conv3d(2 + 8, 32, kernel_size=4, stride=2, padding=1),  # 16x16x16
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),  # 8x8x8
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),  # 4x4x4
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(128, 256, kernel_size=4, stride=1, padding=0),  # 1x1x1
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def _embed_condition(self, condition: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """Embed condition vector spatially."""
        batch_size = condition.size(0)
        
        # Create spatial embedding
        embedding = nn.Linear(self.condition_dim, 8 * np.prod(shape)).to(condition.device)
        embedded = embedding(condition)
        embedded = embedded.view(batch_size, 8, *shape)
        
        return embedded
    
    def generate(self, condition: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Generate holograms from condition."""
        device = condition.device
        
        # Sample noise
        noise = torch.randn(num_samples, self.noise_dim, device=device)
        
        # Expand condition
        if condition.dim() == 1:
            condition = condition.unsqueeze(0).expand(num_samples, -1)
        
        # Combine noise and condition
        input_tensor = torch.cat([noise, condition], dim=1)
        
        # Generate
        generated = self.generator(input_tensor)
        
        return generated


class NeuralHologramGenerator:
    """
    High-level interface for neural hologram generation.
    
    Manages training and inference of neural models for rapid
    acoustic hologram synthesis.
    """
    
    def __init__(
        self,
        model_type: str = "vae",
        config: Optional[TrainingConfig] = None
    ):
        """
        Initialize neural generator.
        
        Args:
            model_type: Type of model ("vae" or "gan")
            config: Training configuration
        """
        self.model_type = model_type
        self.config = config or TrainingConfig()
        
        # Initialize model
        if model_type == "vae":
            self.model = HologramVAE(
                latent_dim=self.config.latent_dim,
                condition_dim=self.config.condition_dim
            )
        elif model_type == "gan":
            self.model = HologramGAN(
                noise_dim=self.config.latent_dim,
                condition_dim=self.config.condition_dim
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.to(self.config.device)
        
        # Training state
        self.training_history = []
        self.is_trained = False
    
    def train(
        self,
        field_patterns: List[np.ndarray],
        conditions: List[Dict[str, Any]],
        validation_split: float = 0.2
    ) -> Dict[str, List[float]]:
        """
        Train the neural model.
        
        Args:
            field_patterns: List of acoustic field patterns
            conditions: List of corresponding conditions
            validation_split: Fraction of data for validation
            
        Returns:
            Training history dictionary
        """
        print(f"Training {self.model_type.upper()} model...")
        
        # Create dataset
        dataset = HologramDataset(field_patterns, conditions)
        
        # Train/validation split
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        
        if self.model_type == "vae":
            history = self._train_vae(train_loader, val_loader)
        else:
            history = self._train_gan(train_loader, val_loader)
        
        self.is_trained = True
        self.training_history = history
        
        return history
    
    def _train_vae(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """Train VAE model."""
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_idx, (data, condition) in enumerate(train_loader):
                data = data.to(self.config.device)
                condition = condition.to(self.config.device)
                
                optimizer.zero_grad()
                
                recon, mu, logvar = self.model(data, condition)
                
                # VAE loss: reconstruction + KL divergence
                recon_loss = F.mse_loss(recon, data, reduction='mean')
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                
                loss = recon_loss + self.config.beta * kl_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for data, condition in val_loader:
                    data = data.to(self.config.device)
                    condition = condition.to(self.config.device)
                    
                    recon, mu, logvar = self.model(data, condition)
                    
                    recon_loss = F.mse_loss(recon, data, reduction='mean')
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    
                    loss = recon_loss + self.config.beta * kl_loss
                    val_loss += loss.item()
            
            # Record losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        return {
            'train_loss': train_losses,
            'val_loss': val_losses
        }
    
    def _train_gan(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """Train GAN model."""
        g_optimizer = optim.Adam(
            self.model.generator.parameters(), 
            lr=self.config.learning_rate, 
            betas=(0.5, 0.999)
        )
        d_optimizer = optim.Adam(
            self.model.discriminator.parameters(), 
            lr=self.config.learning_rate, 
            betas=(0.5, 0.999)
        )
        
        criterion = nn.BCELoss()
        
        g_losses = []
        d_losses = []
        
        for epoch in range(self.config.epochs):
            for batch_idx, (real_data, condition) in enumerate(train_loader):
                batch_size = real_data.size(0)
                real_data = real_data.to(self.config.device)
                condition = condition.to(self.config.device)
                
                # Train Discriminator
                d_optimizer.zero_grad()
                
                # Real samples
                condition_embed = self.model._embed_condition(condition, real_data.shape[2:])
                real_input = torch.cat([real_data, condition_embed], dim=1)
                real_output = self.model.discriminator(real_input)
                real_labels = torch.ones(batch_size, 1, device=self.config.device)
                d_loss_real = criterion(real_output, real_labels)
                
                # Fake samples
                fake_data = self.model.generate(condition, batch_size)
                fake_embed = self.model._embed_condition(condition, fake_data.shape[2:])
                fake_input = torch.cat([fake_data.detach(), fake_embed], dim=1)
                fake_output = self.model.discriminator(fake_input)
                fake_labels = torch.zeros(batch_size, 1, device=self.config.device)
                d_loss_fake = criterion(fake_output, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                
                fake_data = self.model.generate(condition, batch_size)
                fake_embed = self.model._embed_condition(condition, fake_data.shape[2:])
                fake_input = torch.cat([fake_data, fake_embed], dim=1)
                fake_output = self.model.discriminator(fake_input)
                
                g_loss = criterion(fake_output, real_labels)  # Want discriminator to think fake is real
                g_loss.backward()
                g_optimizer.step()
                
                # Record losses
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: G Loss = {np.mean(g_losses[-len(train_loader):]):.4f}, "
                      f"D Loss = {np.mean(d_losses[-len(train_loader):]):.4f}")
        
        return {
            'generator_loss': g_losses,
            'discriminator_loss': d_losses
        }
    
    def generate_hologram(self, condition: Dict[str, Any]) -> np.ndarray:
        """
        Generate hologram for given condition.
        
        Args:
            condition: Condition dictionary specifying desired field
            
        Returns:
            Generated acoustic field pattern
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before generation")
        
        self.model.eval()
        
        # Convert condition to vector
        dataset = HologramDataset([], [])
        condition_vector = dataset._vectorize_condition(condition)
        condition_vector = condition_vector.unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            if self.model_type == "vae":
                generated = self.model.generate(condition_vector)
            else:
                generated = self.model.generate(condition_vector)
            
            # Convert back to complex field
            generated = generated.cpu().numpy()[0]  # Remove batch dimension
            
            # Denormalize
            amplitude = generated[0]  # Amplitude channel
            phase = generated[1] * np.pi  # Phase channel
            
            # Reconstruct complex field
            field = amplitude * np.exp(1j * phase)
        
        return field
    
    def generate_variations(
        self,
        base_condition: Dict[str, Any],
        num_variations: int = 5,
        diversity: float = 0.1
    ) -> List[np.ndarray]:
        """
        Generate variations of a base condition.
        
        Args:
            base_condition: Base condition to vary
            num_variations: Number of variations to generate
            diversity: Amount of variation (0-1)
            
        Returns:
            List of generated field variations
        """
        variations = []
        
        for i in range(num_variations):
            # Add noise to base condition
            varied_condition = base_condition.copy()
            
            if 'focal_points' in varied_condition:
                for fp in varied_condition['focal_points']:
                    # Add position noise
                    pos_noise = np.random.normal(0, diversity * 0.01, 3)  # 1cm std at diversity=1
                    fp['position'] = [p + n for p, n in zip(fp['position'], pos_noise)]
                    
                    # Add pressure noise
                    pressure_noise = np.random.normal(0, diversity * fp.get('pressure', 1000) * 0.1)
                    fp['pressure'] = max(0, fp.get('pressure', 1000) + pressure_noise)
            
            # Generate hologram
            field = self.generate_hologram(varied_condition)
            variations.append(field)
        
        return variations
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        checkpoint = {
            'model_type': self.model_type,
            'config': self.config.__dict__,
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        
        self.model_type = checkpoint['model_type']
        self.config = TrainingConfig(**checkpoint['config'])
        self.training_history = checkpoint['training_history']
        self.is_trained = checkpoint['is_trained']
        
        # Reinitialize model with loaded config
        if self.model_type == "vae":
            self.model = HologramVAE(
                latent_dim=self.config.latent_dim,
                condition_dim=self.config.condition_dim
            )
        else:
            self.model = HologramGAN(
                noise_dim=self.config.latent_dim,
                condition_dim=self.config.condition_dim
            )
        
        self.model.to(self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {filepath}")
    
    @classmethod
    def from_pretrained(cls, name: str) -> 'NeuralHologramGenerator':
        """
        Load a pretrained model.
        
        Args:
            name: Name of pretrained model
            
        Returns:
            Loaded neural generator
        """
        # This would load from a model zoo or repository
        pretrained_models = {
            "acousto-gen-base": "models/acousto_gen_base_vae.pth",
            "acousto-gen-haptics": "models/acousto_gen_haptics_vae.pth",
            "acousto-gen-levitation": "models/acousto_gen_levitation_gan.pth"
        }
        
        if name not in pretrained_models:
            raise ValueError(f"Unknown pretrained model: {name}")
        
        generator = cls()
        # generator.load_model(pretrained_models[name])  # Would load from actual path
        
        print(f"Loaded pretrained model: {name}")
        return generator
    
    def fine_tune(
        self,
        field_patterns: List[np.ndarray],
        conditions: List[Dict[str, Any]],
        epochs: int = 50,
        learning_rate: float = 1e-5
    ) -> Dict[str, List[float]]:
        """
        Fine-tune pretrained model on new data.
        
        Args:
            field_patterns: New training patterns
            conditions: New training conditions
            epochs: Number of fine-tuning epochs
            learning_rate: Lower learning rate for fine-tuning
            
        Returns:
            Fine-tuning history
        """
        if not self.is_trained:
            raise RuntimeError("Model must be pretrained before fine-tuning")
        
        # Save original config
        original_epochs = self.config.epochs
        original_lr = self.config.learning_rate
        
        # Set fine-tuning parameters
        self.config.epochs = epochs
        self.config.learning_rate = learning_rate
        
        print(f"Fine-tuning model with {len(field_patterns)} patterns...")
        
        # Run training
        history = self.train(field_patterns, conditions, validation_split=0.1)
        
        # Restore original config
        self.config.epochs = original_epochs
        self.config.learning_rate = original_lr
        
        return history