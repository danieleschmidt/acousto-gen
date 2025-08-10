"""
Hologram optimization algorithms for acoustic field generation.
Implements gradient-based, genetic, and neural optimization methods.
"""

# Robust dependency handling with graceful degradation
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    # Will be handled by mock system below
    pass
from typing import Optional, Dict, Any, Callable, Tuple, List
from dataclasses import dataclass
import time

# Add path for acousto_gen types
import sys
from pathlib import Path
acousto_gen_path = Path(__file__).parent.parent.parent / "acousto_gen"
sys.path.insert(0, str(acousto_gen_path))

try:
    from type_compat import Tensor, Array
except ImportError:
    # Fallback to Any if types module not available
    Tensor = Any
    Array = Any

# Set up mock backend if needed
try:
    # Check if torch is properly imported
    torch.cuda.is_available
except (NameError, AttributeError):
    # Need to set up mock backend
    try:
        from mock_backend import setup_mock_dependencies
        setup_mock_dependencies()
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        pass


@dataclass
class OptimizationResult:
    """Results from hologram optimization."""
    phases: np.ndarray
    amplitudes: Optional[np.ndarray]
    final_loss: float
    iterations: int
    time_elapsed: float
    convergence_history: List[float]
    metadata: Dict[str, Any]


class GradientOptimizer:
    """
    Gradient-based optimization using automatic differentiation.
    Supports various loss functions and regularization terms.
    """
    
    def __init__(
        self,
        num_elements: int,
        device: str = "cpu",
        learning_rate: float = 0.01,
        optimizer_type: str = "adam"
    ):
        """
        Initialize gradient optimizer.
        
        Args:
            num_elements: Number of transducer elements
            device: Computation device ('cpu' or 'cuda')
            learning_rate: Learning rate for optimization
            optimizer_type: Type of optimizer ('adam', 'sgd', 'lbfgs')
        """
        self.num_elements = num_elements
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        
        # Initialize phase variables
        self.phases = torch.nn.Parameter(
            torch.zeros(num_elements, dtype=torch.float32, device=self.device)
        )
        
        # Setup optimizer
        self._setup_optimizer()
    
    def _setup_optimizer(self):
        """Setup PyTorch optimizer."""
        if self.optimizer_type == "adam":
            self.optimizer = optim.Adam([self.phases], lr=self.learning_rate)
        elif self.optimizer_type == "sgd":
            self.optimizer = optim.SGD([self.phases], lr=self.learning_rate, momentum=0.9)
        elif self.optimizer_type == "lbfgs":
            self.optimizer = optim.LBFGS([self.phases], lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
    
    def optimize(
        self,
        forward_model: Callable,
        target_field: Tensor,
        iterations: int = 1000,
        loss_function: Optional[Callable] = None,
        regularization: Optional[Dict[str, float]] = None,
        callback: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Optimize phases to match target field.
        
        Args:
            forward_model: Function mapping phases to field
            target_field: Desired pressure field
            iterations: Number of optimization iterations
            loss_function: Custom loss function (defaults to MSE)
            regularization: Regularization terms and weights
            callback: Optional callback function for monitoring
            
        Returns:
            OptimizationResult with optimized phases
        """
        start_time = time.time()
        convergence_history = []
        
        # Default loss function
        if loss_function is None:
            loss_function = nn.MSELoss()
        
        # Optimization loop
        for iteration in range(iterations):
            def closure():
                self.optimizer.zero_grad()
                
                # Forward pass
                generated_field = forward_model(self.phases)
                
                # Main loss
                loss = loss_function(generated_field, target_field)
                
                # Add regularization
                if regularization:
                    if "smoothness" in regularization:
                        # Penalize large phase differences
                        phase_diff = torch.diff(self.phases)
                        smooth_loss = regularization["smoothness"] * torch.mean(phase_diff ** 2)
                        loss += smooth_loss
                    
                    if "sparsity" in regularization:
                        # L1 regularization for sparsity
                        sparse_loss = regularization["sparsity"] * torch.mean(torch.abs(self.phases))
                        loss += sparse_loss
                    
                    if "power" in regularization:
                        # Limit total power
                        power_loss = regularization["power"] * torch.mean(self.phases ** 2)
                        loss += power_loss
                
                loss.backward()
                return loss
            
            # Optimization step
            if self.optimizer_type == "lbfgs":
                loss = self.optimizer.step(closure)
            else:
                loss = closure()
                self.optimizer.step()
                
                # Clip phases to valid range
                with torch.no_grad():
                    self.phases.data = torch.remainder(self.phases.data, 2 * np.pi)
            
            # Record convergence
            loss_value = loss.item()
            convergence_history.append(loss_value)
            
            # Callback for monitoring
            if callback and iteration % 10 == 0:
                callback(iteration, loss_value, self.phases.detach().cpu().numpy())
            
            # Early stopping
            if len(convergence_history) > 100:
                recent_improvement = abs(convergence_history[-100] - loss_value)
                if recent_improvement < 1e-6:
                    print(f"Early stopping at iteration {iteration}")
                    break
        
        # Prepare results
        time_elapsed = time.time() - start_time
        
        return OptimizationResult(
            phases=self.phases.detach().cpu().numpy(),
            amplitudes=None,
            final_loss=convergence_history[-1],
            iterations=len(convergence_history),
            time_elapsed=time_elapsed,
            convergence_history=convergence_history,
            metadata={
                "optimizer": self.optimizer_type,
                "learning_rate": self.learning_rate
            }
        )


class GeneticOptimizer:
    """
    Genetic algorithm for hologram optimization.
    Useful for non-differentiable objectives and global optimization.
    """
    
    def __init__(
        self,
        num_elements: int,
        population_size: int = 100,
        elite_fraction: float = 0.1,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7
    ):
        """
        Initialize genetic optimizer.
        
        Args:
            num_elements: Number of transducer elements
            population_size: Size of population
            elite_fraction: Fraction of population to keep as elite
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.num_elements = num_elements
        self.population_size = population_size
        self.elite_size = int(population_size * elite_fraction)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Initialize population
        self.population = np.random.uniform(
            0, 2 * np.pi, (population_size, num_elements)
        )
        self.fitness_scores = np.zeros(population_size)
    
    def _evaluate_fitness(
        self,
        forward_model: Callable,
        target_field: np.ndarray
    ):
        """Evaluate fitness of entire population."""
        for i, individual in enumerate(self.population):
            generated = forward_model(individual)
            # Fitness is negative of error (higher is better)
            error = np.mean(np.abs(generated - target_field) ** 2)
            self.fitness_scores[i] = -error
    
    def _selection(self) -> np.ndarray:
        """Tournament selection for parent selection."""
        tournament_size = 3
        selected = []
        
        for _ in range(self.population_size):
            # Random tournament
            candidates = np.random.choice(
                self.population_size, tournament_size, replace=False
            )
            winner = candidates[np.argmax(self.fitness_scores[candidates])]
            selected.append(self.population[winner])
        
        return np.array(selected)
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover between two parents."""
        if np.random.random() < self.crossover_rate:
            mask = np.random.random(self.num_elements) < 0.5
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        
        return child1, child2
    
    def _mutation(self, individual: np.ndarray) -> np.ndarray:
        """Gaussian mutation of phases."""
        mask = np.random.random(self.num_elements) < self.mutation_rate
        individual[mask] += np.random.normal(0, 0.5, np.sum(mask))
        individual = np.remainder(individual, 2 * np.pi)
        return individual
    
    def optimize(
        self,
        forward_model: Callable,
        target_field: np.ndarray,
        generations: int = 100,
        callback: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Optimize using genetic algorithm.
        
        Args:
            forward_model: Function mapping phases to field
            target_field: Desired pressure field
            generations: Number of generations
            callback: Optional callback for monitoring
            
        Returns:
            OptimizationResult with best solution
        """
        start_time = time.time()
        convergence_history = []
        
        for generation in range(generations):
            # Evaluate fitness
            self._evaluate_fitness(forward_model, target_field)
            
            # Record best fitness
            best_fitness = np.max(self.fitness_scores)
            convergence_history.append(-best_fitness)  # Convert back to error
            
            # Sort by fitness
            sorted_indices = np.argsort(self.fitness_scores)[::-1]
            self.population = self.population[sorted_indices]
            self.fitness_scores = self.fitness_scores[sorted_indices]
            
            # Keep elite
            new_population = self.population[:self.elite_size].copy()
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Select parents
                parents = self._selection()
                idx1, idx2 = np.random.choice(len(parents), 2, replace=False)
                
                # Crossover
                child1, child2 = self._crossover(parents[idx1], parents[idx2])
                
                # Mutation
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)
                
                new_population = np.vstack([new_population, child1])
                if len(new_population) < self.population_size:
                    new_population = np.vstack([new_population, child2])
            
            self.population = new_population[:self.population_size]
            
            # Callback
            if callback and generation % 10 == 0:
                callback(generation, -best_fitness, self.population[0])
        
        # Final evaluation
        self._evaluate_fitness(forward_model, target_field)
        best_idx = np.argmax(self.fitness_scores)
        
        time_elapsed = time.time() - start_time
        
        return OptimizationResult(
            phases=self.population[best_idx],
            amplitudes=None,
            final_loss=-self.fitness_scores[best_idx],
            iterations=generations,
            time_elapsed=time_elapsed,
            convergence_history=convergence_history,
            metadata={
                "population_size": self.population_size,
                "final_fitness": self.fitness_scores[best_idx]
            }
        )


class NeuralHologramGenerator(nn.Module):
    """
    Neural network for rapid hologram generation.
    Learns mapping from target fields to phase patterns.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[int] = [512, 256, 128],
        activation: str = "relu"
    ):
        """
        Initialize neural hologram generator.
        
        Args:
            input_size: Size of input (flattened target field)
            output_size: Number of transducer elements
            hidden_layers: List of hidden layer sizes
            activation: Activation function type
        """
        super().__init__()
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.1))
            elif activation == "elu":
                layers.append(nn.ELU())
            
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Tanh())  # Output in [-1, 1], scale to phases later
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, target_field: Tensor) -> Tensor:
        """
        Generate phase pattern from target field.
        
        Args:
            target_field: Batch of target fields
            
        Returns:
            Phase patterns in radians
        """
        # Flatten input
        batch_size = target_field.shape[0]
        flattened = target_field.view(batch_size, -1)
        
        # Generate phases
        output = self.network(flattened)
        
        # Scale to [0, 2π]
        phases = (output + 1) * np.pi
        
        return phases
    
    def train_generator(
        self,
        training_data: List[Tuple[np.ndarray, np.ndarray]],
        validation_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: str = "cpu"
    ) -> Dict[str, List[float]]:
        """
        Train the neural generator on field-phase pairs.
        
        Args:
            training_data: List of (target_field, phase_pattern) pairs
            validation_data: Optional validation set
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            device: Training device
            
        Returns:
            Training history
        """
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        # Prepare data
        fields = torch.tensor(np.array([d[0] for d in training_data]), dtype=torch.float32)
        phases = torch.tensor(np.array([d[1] for d in training_data]), dtype=torch.float32)
        
        dataset = torch.utils.data.TensorDataset(fields, phases)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )
        
        history = {"train_loss": [], "val_loss": []}
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.train()
            train_losses = []
            
            for batch_fields, batch_phases in dataloader:
                batch_fields = batch_fields.to(device)
                batch_phases = batch_phases.to(device)
                
                optimizer.zero_grad()
                predicted = self(batch_fields)
                loss = criterion(predicted, batch_phases)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history["train_loss"].append(avg_train_loss)
            
            # Validation
            if validation_data:
                self.eval()
                val_fields = torch.tensor(
                    np.array([d[0] for d in validation_data]),
                    dtype=torch.float32,
                    device=device
                )
                val_phases = torch.tensor(
                    np.array([d[1] for d in validation_data]),
                    dtype=torch.float32,
                    device=device
                )
                
                with torch.no_grad():
                    val_predicted = self(val_fields)
                    val_loss = criterion(val_predicted, val_phases)
                
                history["val_loss"].append(val_loss.item())
                scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}")
        
        return history
    
    def generate(
        self,
        target_field: np.ndarray,
        device: str = "cpu"
    ) -> np.ndarray:
        """
        Generate phase pattern for a target field.
        
        Args:
            target_field: Desired pressure field
            device: Computation device
            
        Returns:
            Phase pattern in radians
        """
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        
        field_tensor = torch.tensor(
            target_field, dtype=torch.float32, device=device
        ).unsqueeze(0)
        
        with torch.no_grad():
            phases = self(field_tensor)
        
        return phases.squeeze().cpu().numpy()