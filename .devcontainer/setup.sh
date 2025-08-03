#!/bin/bash
# Acousto-Gen Development Environment Setup Script

set -euo pipefail

echo "🚀 Setting up Acousto-Gen development environment..."

# Create directories
mkdir -p /workspace/data /workspace/logs /workspace/cache /workspace/notebooks

# Create Python virtual environment
echo "📦 Creating Python virtual environment..."
python3.11 -m venv /workspace/.venv
source /workspace/.venv/bin/activate

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install Acousto-Gen in development mode
echo "🔧 Installing Acousto-Gen in development mode..."
cd /workspace
pip install -e ".[dev,hardware,full]"

# Install additional development tools
pip install \
    jupyter \
    jupyterlab \
    ipywidgets \
    notebook \
    tensorboard \
    wandb \
    nvitop \
    memory-profiler \
    line-profiler \
    pre-commit

# Setup pre-commit hooks
echo "🔒 Setting up pre-commit hooks..."
pre-commit install

# Setup Jupyter extensions
echo "📓 Setting up Jupyter extensions..."
jupyter lab build

# Create development configuration
echo "⚙️ Creating development configuration..."
mkdir -p /workspace/config
cat > /workspace/config/development.yaml << 'EOF'
environment: development

logging:
  level: DEBUG
  format: detailed
  console: true
  file: logs/acousto-gen-dev.log

system:
  gpu_memory_limit: "4GB"
  auto_reload: true
  debug_mode: true
  worker_processes: 1

optimization:
  default_iterations: 100
  convergence_threshold: 1e-4
  timeout: 60

hardware:
  simulation_mode: true
  safety_checks: relaxed
  auto_detect: false

database:
  url: "sqlite:///data/acousto-gen-dev.db"
  echo: true

cache:
  backend: "memory"
  ttl: 300

monitoring:
  prometheus_enabled: false
  metrics_port: 9090
EOF

# Create .env file for development
echo "🔐 Creating environment file..."
cat > /workspace/.env << 'EOF'
# Acousto-Gen Development Environment
ACOUSTO_GEN_ENV=development
ACOUSTO_GEN_CONFIG=/workspace/config/development.yaml
PYTHONPATH=/workspace/src:/workspace
CUDA_VISIBLE_DEVICES=all

# Development settings
DEBUG=true
LOG_LEVEL=DEBUG
AUTO_RELOAD=true

# Database
DATABASE_URL=sqlite:///data/acousto-gen-dev.db

# Cache
CACHE_BACKEND=memory
CACHE_TTL=300

# Optional: API keys for development
# WANDB_API_KEY=your_wandb_key_here
# GITHUB_TOKEN=your_github_token_here
EOF

# Setup Git configuration for development
echo "📋 Setting up Git configuration..."
git config --global user.name "Acousto-Gen Developer"
git config --global user.email "dev@acousto-gen.local"
git config --global init.defaultBranch main

# Create sample data and notebooks
echo "📊 Creating sample data and notebooks..."

# Sample notebook
mkdir -p /workspace/notebooks
cat > /workspace/notebooks/getting_started.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Acousto-Gen Getting Started\n",
    "\n",
    "Welcome to the Acousto-Gen development environment! This notebook will help you get started with acoustic holography development."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import torch\n",
    "\n",
    "# Import Acousto-Gen modules\n",
    "from src.physics.transducers.transducer_array import UltraLeap256\n",
    "from src.physics.propagation.wave_propagator import WavePropagator\n",
    "from src.models.acoustic_field import create_focus_target\n",
    "from src.optimization.hologram_optimizer import GradientOptimizer\n",
    "\n",
    "print(\"Acousto-Gen development environment ready!\")\n",
    "print(f\"PyTorch version: {torch.__version__}\")\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"CUDA device: {torch.cuda.get_device_name()}\")\n",
    "    print(f\"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a simple acoustic hologram\n",
    "array = UltraLeap256()\n",
    "propagator = WavePropagator(device=\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "optimizer = GradientOptimizer(num_elements=256)\n",
    "\n",
    "# Create target focus point\n",
    "target = create_focus_target(\n",
    "    position=[0, 0, 0.1],  # 10cm above array\n",
    "    pressure=3000,  # Pa\n",
    "    shape=(50, 50, 50)\n",
    ")\n",
    "\n",
    "print(f\"Array: {array.name} with {len(array.elements)} elements\")\n",
    "print(f\"Target field shape: {target.shape}\")\n",
    "print(f\"Target peak pressure: {np.max(target.get_amplitude_field()):.0f} Pa\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize the target field\n",
    "x_coords, y_coords, field_slice = target.get_slice(plane=\"xy\", position=0.1)\n",
    "\n",
    "plt.figure(figsize=(10, 8))\n",
    "plt.imshow(\n",
    "    np.abs(field_slice).T,\n",
    "    extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],\n",
    "    origin='lower',\n",
    "    cmap='viridis'\n",
    ")\n",
    "plt.colorbar(label='Pressure (Pa)')\n",
    "plt.xlabel('X (m)')\n",
    "plt.ylabel('Y (m)')\n",
    "plt.title('Target Acoustic Field (XY plane at z=0.1m)')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Set permissions
chmod +x /workspace/.devcontainer/setup.sh
chmod +x /workspace/.devcontainer/startup.sh 2>/dev/null || true

echo "✅ Acousto-Gen development environment setup complete!"
echo ""
echo "🔧 Next steps:"
echo "  1. Open a terminal and run: source /workspace/.venv/bin/activate"
echo "  2. Start Jupyter Lab: jupyter lab --ip=0.0.0.0 --port=8888 --allow-root"
echo "  3. Start the API server: acousto-gen serve --reload --debug"
echo "  4. Open the getting started notebook: notebooks/getting_started.ipynb"
echo ""
echo "📚 Documentation:"
echo "  - Development guide: DEVELOPMENT.md"
echo "  - API docs: http://localhost:8000/docs"
echo "  - Jupyter Lab: http://localhost:8888"