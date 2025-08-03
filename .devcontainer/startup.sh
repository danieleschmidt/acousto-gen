#!/bin/bash
# Acousto-Gen Development Environment Startup Script

set -euo pipefail

echo "🚀 Starting Acousto-Gen development services..."

# Activate virtual environment
source /workspace/.venv/bin/activate

# Check system resources
echo "💻 System Resources:"
echo "  CPU cores: $(nproc)"
echo "  Memory: $(free -h | grep Mem | awk '{print $2}')"
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "  GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
fi

# Check Python environment
echo ""
echo "🐍 Python Environment:"
echo "  Python: $(python --version)"
echo "  Virtual env: $VIRTUAL_ENV"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Verify Acousto-Gen installation
echo ""
echo "🔊 Acousto-Gen Status:"
if python -c "import acousto_gen; print(f'Version: {acousto_gen.__version__}')" 2>/dev/null; then
    echo "  ✅ Acousto-Gen installed successfully"
else
    echo "  ⚠️  Acousto-Gen not found, installing..."
    pip install -e ".[dev,hardware,full]"
fi

# Run basic health checks
echo ""
echo "🔍 Health Checks:"

# Test imports
python -c "
try:
    from src.physics.transducers.transducer_array import UltraLeap256
    from src.physics.propagation.wave_propagator import WavePropagator
    from src.optimization.hologram_optimizer import GradientOptimizer
    print('  ✅ Core modules import successfully')
except Exception as e:
    print(f'  ❌ Import error: {e}')
"

# Test GPU
python -c "
import torch
if torch.cuda.is_available():
    try:
        x = torch.tensor([1.0]).cuda()
        print('  ✅ GPU computation working')
    except Exception as e:
        print(f'  ⚠️  GPU issue: {e}')
else:
    print('  ℹ️  Running in CPU mode')
"

# Start background services
echo ""
echo "🔧 Starting services..."

# Start Prometheus metrics (if enabled)
if [ "${PROMETHEUS_ENABLED:-false}" = "true" ]; then
    echo "  Starting Prometheus metrics server..."
    python -c "
from src.monitoring.metrics import MetricsCollector
collector = MetricsCollector()
collector.start_collection()
" &
fi

# Create useful aliases
echo ""
echo "📋 Creating helpful aliases..."
cat >> ~/.bashrc << 'EOF'

# Acousto-Gen aliases
alias ag='acousto-gen'
alias agserve='acousto-gen serve --reload --debug --host 0.0.0.0'
alias agjupyter='jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser'
alias agtest='pytest tests/ -v'
alias agcov='pytest tests/ --cov=src --cov-report=html'
alias aglint='ruff check src/ tests/'
alias agformat='ruff format src/ tests/'
alias agdocs='cd docs && make html && make serve'

# Development shortcuts
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'

# Git shortcuts
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline -10'
alias gd='git diff'

# Python shortcuts
alias py='python'
alias ipy='ipython'
alias nb='jupyter notebook'
alias lab='jupyter lab'

EOF

source ~/.bashrc

echo ""
echo "✅ Acousto-Gen development environment ready!"
echo ""
echo "🔧 Available commands:"
echo "  ag serve --reload     # Start API server with hot reload"
echo "  ag test              # Run test suite"
echo "  agjupyter           # Start Jupyter Lab"
echo "  agtest              # Run tests"
echo "  agcov               # Run tests with coverage"
echo "  aglint              # Check code quality"
echo "  agformat            # Format code"
echo ""
echo "🌐 Service URLs:"
echo "  API Server: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "  Jupyter Lab: http://localhost:8888"
if [ "${PROMETHEUS_ENABLED:-false}" = "true" ]; then
    echo "  Metrics: http://localhost:9090"
fi
echo ""
echo "Happy coding! 🎵🔊"