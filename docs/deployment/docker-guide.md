# Docker Deployment Guide

This guide covers containerized deployment of Acousto-Gen using Docker and Docker Compose.

## Quick Start

### Basic Development Setup

```bash
# Clone repository
git clone https://github.com/danieleschmidt/acousto-gen.git
cd acousto-gen

# Start development environment
docker-compose up acousto-gen-dev

# Access the container
docker-compose exec acousto-gen-dev bash
```

### Jupyter Notebook Environment

```bash
# Start Jupyter Lab
docker-compose up jupyter

# Access at http://localhost:8888
# Token: acousto-gen-dev-token
```

### GPU-Accelerated Environment

```bash
# Start GPU-enabled container (requires NVIDIA Docker)
docker-compose --profile gpu up acousto-gen-gpu
```

## Docker Images

### Available Targets

The Dockerfile uses multi-stage builds to create optimized images for different use cases:

| Target | Description | Use Case |
|--------|-------------|----------|
| `base` | Base system with Python and CUDA | Foundation layer |
| `dependencies` | Python dependencies installed | Reusable dependency layer |
| `development` | Full development environment | Interactive development |
| `testing` | Testing environment with test dependencies | CI/CD testing |
| `production` | Minimal production image | Production deployment |
| `gpu` | GPU-optimized production image | GPU workloads |
| `jupyter` | Jupyter Lab environment | Interactive notebooks |

### Building Images

```bash
# Build development image
docker build --target development -t acousto-gen:dev .

# Build production image
docker build --target production -t acousto-gen:prod .

# Build GPU image
docker build --target gpu -t acousto-gen:gpu .

# Build with BuildKit for better caching
DOCKER_BUILDKIT=1 docker build --target development -t acousto-gen:dev .
```

## Service Profiles

Docker Compose uses profiles to organize services:

### Default Services

Run automatically with `docker-compose up`:

- `acousto-gen-dev`: Development environment
- `jupyter`: Jupyter Lab server

### Optional Profiles

Activated with `--profile` flag:

```bash
# GPU services
docker-compose --profile gpu up

# Testing services  
docker-compose --profile test up

# Production services
docker-compose --profile production up

# Documentation services
docker-compose --profile docs up

# Database services
docker-compose --profile database up

# Caching services
docker-compose --profile cache up

# Monitoring services
docker-compose --profile monitoring up
```

### Combined Profiles

```bash
# Full development stack with database and monitoring
docker-compose --profile database --profile monitoring up
```

## Environment Configuration

### Environment Variables

Key environment variables for configuration:

```bash
# Device selection
ACOUSTO_DEVICE=cuda          # or 'cpu'

# Logging
ACOUSTO_LOG_LEVEL=INFO       # DEBUG, INFO, WARNING, ERROR

# Safety
ACOUSTO_ENABLE_SAFETY=true   # Enable safety monitoring

# CUDA (for GPU containers)
CUDA_VISIBLE_DEVICES=0       # GPU device selection
NVIDIA_VISIBLE_DEVICES=all   # NVIDIA Docker setting
```

### Configuration Files

Mount configuration files as volumes:

```yaml
volumes:
  - ./config.yaml:/home/acousto/config.yaml:ro
```

## Volume Management

### Persistent Volumes

| Volume | Purpose | Data |
|--------|---------|------|
| `acousto-gen-data` | Application data | Results, models, cache |
| `acousto-gen-db-data` | Database storage | Experiment tracking |
| `acousto-gen-venv` | Python virtual env | Development dependencies |

### Backup Volumes

```bash
# Backup data volume
docker run --rm -v acousto-gen-data:/data -v $(pwd):/backup \
  ubuntu tar czf /backup/acousto-data-backup.tar.gz /data

# Restore data volume
docker run --rm -v acousto-gen-data:/data -v $(pwd):/backup \
  ubuntu tar xzf /backup/acousto-data-backup.tar.gz -C /
```

## Production Deployment

### Single Container Deployment

```bash
# Build production image
docker build --target production -t acousto-gen:latest .

# Run production container
docker run -d \
  --name acousto-gen-prod \
  --restart unless-stopped \
  -v acousto-data:/home/acousto/data \
  -e ACOUSTO_LOG_LEVEL=INFO \
  -e ACOUSTO_ENABLE_SAFETY=true \
  acousto-gen:latest
```

### Docker Compose Production

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  acousto-gen:
    image: acousto-gen:latest
    restart: unless-stopped
    volumes:
      - acousto-data:/home/acousto/data
    environment:
      - ACOUSTO_LOG_LEVEL=INFO
      - ACOUSTO_ENABLE_SAFETY=true
    ports:
      - "8080:8080"
    
volumes:
  acousto-data:
    driver: local
```

```bash
# Deploy production stack
docker-compose -f docker-compose.prod.yml up -d
```

## GPU Support

### Prerequisites

1. **NVIDIA Driver**: Compatible GPU driver installed
2. **NVIDIA Docker**: Docker runtime for GPU access
3. **Docker Compose**: Version 1.28+ for GPU support

### Installation

```bash
# Install NVIDIA Docker (Ubuntu)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Verification

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Test with Acousto-Gen
docker-compose --profile gpu up acousto-gen-gpu
docker-compose exec acousto-gen-gpu python -c "import torch; print(torch.cuda.is_available())"
```

## Development Workflow

### Interactive Development

```bash
# Start development environment
docker-compose up -d acousto-gen-dev

# Enter container
docker-compose exec acousto-gen-dev bash

# Run tests
pytest tests/

# Start Jupyter in container
acousto-gen-dev:~$ jupyter lab --ip=0.0.0.0 --port=8888
```

### Code Synchronization

Development container mounts source code:

```yaml
volumes:
  - .:/home/acousto/work:rw  # Live code sync
```

Changes on host are immediately reflected in container.

### Hot Reloading

For development servers with hot reload:

```bash
# In container
watchdog --patterns="*.py" --recursive acousto_gen/ \
  --command="python -m acousto_gen.server --reload"
```

## Testing in Docker

### Unit Tests

```bash
# Run test suite
docker-compose --profile test up acousto-gen-test

# Run specific tests
docker-compose run --rm acousto-gen-test pytest tests/unit/ -v

# Run with coverage
docker-compose run --rm acousto-gen-test \
  pytest --cov=acousto_gen --cov-report=html
```

### Integration Tests

```bash
# Run integration tests with database
docker-compose --profile database --profile test up -d postgres
docker-compose run --rm acousto-gen-test pytest tests/integration/ -v
```

### Performance Tests

```bash
# Run GPU performance tests
docker-compose --profile gpu run --rm acousto-gen-gpu \
  pytest tests/performance/ -m gpu -v
```

## Monitoring and Observability

### Prometheus + Grafana Stack

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d prometheus grafana

# Access Grafana at http://localhost:3000
# Username: admin, Password: acousto_admin
```

### Health Checks

Built-in health checks for containers:

```bash
# Check container health
docker-compose ps

# View health check logs
docker inspect acousto-gen-prod | jq '.[0].State.Health'
```

### Log Management

```bash
# View logs
docker-compose logs -f acousto-gen-dev

# View specific service logs
docker-compose logs acousto-gen-prod

# Export logs
docker-compose logs --no-color acousto-gen-prod > acousto.log
```

## Security Best Practices

### User Security

- Containers run as non-root user `acousto`
- Minimal attack surface in production images
- No unnecessary packages in production

### Network Security

```yaml
# Restrict network access
networks:
  acousto-net:
    driver: bridge
    internal: true  # No external access
```

### Secrets Management

```bash
# Use Docker secrets for sensitive data
echo "secret_value" | docker secret create acousto_secret -

# Mount in compose
services:
  acousto-gen:
    secrets:
      - acousto_secret
```

## Troubleshooting

### Common Issues

#### CUDA Not Available

```bash
# Check NVIDIA Docker installation
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Verify compose GPU config
docker-compose config --services
```

#### Permission Issues

```bash
# Fix volume permissions
docker-compose exec acousto-gen-dev sudo chown -R acousto:acousto /home/acousto/
```

#### Memory Issues

```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory

# Monitor memory usage
docker stats acousto-gen-dev
```

### Debugging

```bash
# Enter container for debugging
docker-compose exec acousto-gen-dev bash

# Debug specific service
docker-compose run --rm --entrypoint bash acousto-gen-dev

# View container resource usage
docker stats

# Inspect container
docker inspect acousto-gen-dev
```

### Performance Optimization

#### Build Optimization

```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Multi-platform builds
docker buildx build --platform linux/amd64,linux/arm64 .

# Build cache optimization
docker build --cache-from acousto-gen:dev .
```

#### Runtime Optimization

```yaml
# Optimize resource limits
services:
  acousto-gen:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
```

## Maintenance

### Regular Updates

```bash
# Update base images
docker-compose pull

# Rebuild with latest dependencies
docker-compose build --no-cache

# Prune unused resources
docker system prune -a
```

### Backup Strategy

```bash
# Backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker run --rm \
  -v acousto-gen-data:/data \
  -v $(pwd)/backups:/backup \
  ubuntu tar czf /backup/acousto-backup-$DATE.tar.gz /data
```

### Monitoring Disk Usage

```bash
# Check Docker disk usage
docker system df

# Monitor volume sizes
docker volume ls -q | xargs docker volume inspect | \
  jq -r '.[]|[.Name,.Mountpoint]|@csv'
```

For more advanced deployment scenarios, see the [Kubernetes Guide](kubernetes-guide.md) and [Cloud Deployment Guide](cloud-deployment.md).