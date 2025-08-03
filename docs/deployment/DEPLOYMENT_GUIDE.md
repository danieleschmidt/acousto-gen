# Acousto-Gen Deployment Guide

This guide covers production deployment of Acousto-Gen acoustic holography systems for research, development, and commercial applications.

## Overview

Acousto-Gen supports multiple deployment scenarios:
- **Research Labs**: Interactive development and experimentation
- **Production Systems**: Real-time acoustic manipulation
- **Cloud Services**: Scalable computation and API services
- **Edge Devices**: Embedded systems with hardware integration

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), Windows 10+, macOS 11+
- **CPU**: Intel i5 / AMD Ryzen 5 (4+ cores)
- **RAM**: 8GB system memory
- **Storage**: 10GB available space
- **Python**: 3.9 or higher

### Recommended Requirements
- **OS**: Ubuntu 22.04 LTS (best tested)
- **CPU**: Intel i7 / AMD Ryzen 7 (8+ cores)
- **RAM**: 32GB system memory
- **GPU**: NVIDIA RTX 3080+ (10GB+ VRAM)
- **Storage**: 100GB SSD storage
- **Network**: Gigabit Ethernet

### Production Requirements
- **CPU**: Intel Xeon / AMD EPYC (16+ cores)
- **RAM**: 64GB+ system memory
- **GPU**: NVIDIA A100 (40GB VRAM) or multiple GPUs
- **Storage**: NVMe SSD with high IOPS
- **Network**: 10Gb Ethernet
- **UPS**: Uninterruptible power supply
- **Cooling**: Adequate cooling for GPU workloads

## Deployment Methods

### 1. Docker Deployment (Recommended)

#### Basic Deployment
```bash
# Pull latest image
docker pull acousto-gen/acousto-gen:latest

# Run with GPU support
docker run -d \
  --name acousto-gen \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  acousto-gen/acousto-gen:latest
```

#### Production Deployment with Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  acousto-gen:
    image: acousto-gen/acousto-gen:latest
    container_name: acousto-gen-prod
    restart: unless-stopped
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
    ports:
      - "8000:8000"
      - "8443:8443"  # HTTPS
    
    volumes:
      - ./data:/app/data
      - ./config:/app/config
      - ./logs:/app/logs
      - /dev:/dev  # Hardware access
    
    environment:
      - ACOUSTO_GEN_ENV=production
      - ACOUSTO_GEN_GPU_MEMORY_LIMIT=8GB
      - ACOUSTO_GEN_LOG_LEVEL=INFO
    
    privileged: true  # For hardware access
    
    networks:
      - acousto-gen-net
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  redis:
    image: redis:7-alpine
    container_name: acousto-gen-redis
    restart: unless-stopped
    volumes:
      - redis_data:/data
    networks:
      - acousto-gen-net

  prometheus:
    image: prom/prometheus:latest
    container_name: acousto-gen-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - acousto-gen-net

  grafana:
    image: grafana/grafana:latest
    container_name: acousto-gen-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=acousto-gen-admin
    networks:
      - acousto-gen-net

volumes:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  acousto-gen-net:
    driver: bridge
```

### 2. Kubernetes Deployment

#### Namespace and Configuration
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: acousto-gen
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: acousto-gen-config
  namespace: acousto-gen
data:
  config.yaml: |
    system:
      gpu_memory_limit: "8GB"
      log_level: "INFO"
      safety_limits:
        max_pressure: 4000
        max_temperature: 60
    
    optimization:
      default_method: "gradient"
      max_iterations: 1000
      convergence_threshold: 1e-6
    
    hardware:
      auto_detect: true
      safety_monitoring: true
      emergency_shutdown: true
```

#### Main Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: acousto-gen
  namespace: acousto-gen
  labels:
    app: acousto-gen
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  
  selector:
    matchLabels:
      app: acousto-gen
  
  template:
    metadata:
      labels:
        app: acousto-gen
    spec:
      containers:
      - name: acousto-gen
        image: acousto-gen/acousto-gen:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8443
          name: https
        
        env:
        - name: ACOUSTO_GEN_ENV
          value: "production"
        - name: ACOUSTO_GEN_CONFIG
          value: "/config/config.yaml"
        
        resources:
          requests:
            memory: "8Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
        
        volumeMounts:
        - name: config
          mountPath: /config
        - name: data
          mountPath: /app/data
        - name: hardware
          mountPath: /dev
        
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 1
      
      volumes:
      - name: config
        configMap:
          name: acousto-gen-config
      - name: data
        persistentVolumeClaim:
          claimName: acousto-gen-data
      - name: hardware
        hostPath:
          path: /dev
      
      nodeSelector:
        acousto-gen.io/gpu: "true"
      
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule

---
apiVersion: v1
kind: Service
metadata:
  name: acousto-gen-service
  namespace: acousto-gen
spec:
  selector:
    app: acousto-gen
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: https
    port: 443
    targetPort: 8443
  type: LoadBalancer

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: acousto-gen-data
  namespace: acousto-gen
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
```

### 3. Bare Metal Installation

#### System Preparation
```bash
#!/bin/bash
# install.sh - Production installation script

set -euo pipefail

# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    git \
    curl \
    wget \
    htop \
    nginx \
    redis-server \
    postgresql-14 \
    udev

# Install NVIDIA drivers and CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-2

# Create dedicated user
sudo useradd -m -s /bin/bash acousto-gen
sudo usermod -a -G dialout,video acousto-gen

# Create directories
sudo mkdir -p /opt/acousto-gen/{app,data,logs,config}
sudo chown -R acousto-gen:acousto-gen /opt/acousto-gen

# Switch to acousto-gen user
sudo -u acousto-gen bash << 'EOF'
cd /opt/acousto-gen

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Acousto-Gen
pip install --upgrade pip
pip install acousto-gen[full]

# Verify installation
acousto-gen --version
acousto-gen test-installation
EOF

echo "Installation complete!"
echo "Start service with: sudo systemctl start acousto-gen"
```

#### Systemd Service
```ini
# /etc/systemd/system/acousto-gen.service
[Unit]
Description=Acousto-Gen Acoustic Holography Service
After=network.target
Wants=network.target

[Service]
Type=exec
User=acousto-gen
Group=acousto-gen
WorkingDirectory=/opt/acousto-gen
Environment=PATH=/opt/acousto-gen/venv/bin
Environment=ACOUSTO_GEN_CONFIG=/opt/acousto-gen/config/production.yaml
ExecStart=/opt/acousto-gen/venv/bin/acousto-gen serve --host 0.0.0.0 --port 8000
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=acousto-gen

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/acousto-gen/data /opt/acousto-gen/logs
DeviceAllow=/dev/nvidia0 rw
DeviceAllow=/dev/nvidiactl rw
DeviceAllow=/dev/nvidia-modeset rw
DeviceAllow=/dev/nvidia-uvm rw
DeviceAllow=/dev/ttyUSB* rw

[Install]
WantedBy=multi-user.target
```

## Configuration Management

### Environment-Specific Configurations

#### Development Configuration
```yaml
# config/development.yaml
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

optimization:
  default_iterations: 100
  convergence_threshold: 1e-4

hardware:
  simulation_mode: true
  safety_checks: relaxed
```

#### Production Configuration
```yaml
# config/production.yaml
environment: production

logging:
  level: INFO
  format: json
  console: false
  file: logs/acousto-gen.log
  max_file_size: 100MB
  backup_count: 10

system:
  gpu_memory_limit: "8GB"
  auto_reload: false
  debug_mode: false
  worker_processes: 4

optimization:
  default_iterations: 1000
  convergence_threshold: 1e-6
  timeout: 300

hardware:
  simulation_mode: false
  safety_checks: strict
  monitoring_frequency: 100  # Hz
  emergency_protocols: enabled

security:
  authentication: required
  rate_limiting: enabled
  cors_origins: ["https://lab.example.com"]
  tls_cert: /etc/ssl/certs/acousto-gen.crt
  tls_key: /etc/ssl/private/acousto-gen.key

database:
  url: postgresql://acousto:password@localhost/acousto_gen
  pool_size: 20
  pool_timeout: 30

cache:
  backend: redis
  url: redis://localhost:6379/0
  ttl: 3600

monitoring:
  prometheus_enabled: true
  metrics_port: 9090
  health_check_interval: 30
```

### Configuration Validation
```python
# scripts/validate_config.py
import yaml
from jsonschema import validate, ValidationError

def validate_config(config_file: str):
    """Validate configuration against schema."""
    
    schema = {
        "type": "object",
        "required": ["environment", "logging", "system"],
        "properties": {
            "environment": {"enum": ["development", "staging", "production"]},
            "logging": {
                "type": "object",
                "required": ["level", "format"],
                "properties": {
                    "level": {"enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                    "format": {"enum": ["simple", "detailed", "json"]}
                }
            },
            "system": {
                "type": "object",
                "required": ["gpu_memory_limit"],
                "properties": {
                    "gpu_memory_limit": {"type": "string", "pattern": r"^\d+GB$"}
                }
            }
        }
    }
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    try:
        validate(instance=config, schema=schema)
        print(f"✓ Configuration {config_file} is valid")
        return True
    except ValidationError as e:
        print(f"✗ Configuration error: {e.message}")
        return False

if __name__ == "__main__":
    import sys
    validate_config(sys.argv[1])
```

## Security Considerations

### Network Security
```nginx
# /etc/nginx/sites-available/acousto-gen
server {
    listen 80;
    server_name acousto-gen.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name acousto-gen.example.com;
    
    ssl_certificate /etc/ssl/certs/acousto-gen.crt;
    ssl_certificate_key /etc/ssl/private/acousto-gen.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    location /metrics {
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;
        
        proxy_pass http://127.0.0.1:9090;
    }
}
```

### Authentication and Authorization
```python
# security/auth.py
from functools import wraps
from typing import List, Optional
import jwt
from flask import request, jsonify

class AuthenticationManager:
    """Handle user authentication and authorization."""
    
    def __init__(self, secret_key: str, algorithms: List[str] = None):
        self.secret_key = secret_key
        self.algorithms = algorithms or ["HS256"]
    
    def generate_token(self, user_id: str, permissions: List[str]) -> str:
        """Generate JWT token for user."""
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithms[0])
    
    def verify_token(self, token: str) -> Optional[dict]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=self.algorithms)
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

def require_auth(permissions: List[str] = None):
    """Decorator to require authentication and permissions."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                return jsonify({'error': 'Authentication required'}), 401
            
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
                payload = auth_manager.verify_token(token)
                if not payload:
                    return jsonify({'error': 'Invalid token'}), 401
                
                if permissions:
                    user_permissions = payload.get('permissions', [])
                    if not any(perm in user_permissions for perm in permissions):
                        return jsonify({'error': 'Insufficient permissions'}), 403
                
                request.user = payload
                return f(*args, **kwargs)
            
            except Exception as e:
                return jsonify({'error': 'Authentication failed'}), 401
        
        return decorated_function
    return decorator
```

## Monitoring and Observability

### Prometheus Metrics
```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps

# Define metrics
optimization_requests = Counter('acousto_gen_optimization_requests_total', 
                               'Total optimization requests', ['method', 'status'])

optimization_duration = Histogram('acousto_gen_optimization_duration_seconds',
                                 'Optimization duration in seconds', ['method'])

hardware_status = Gauge('acousto_gen_hardware_status', 
                       'Hardware connection status', ['device'])

gpu_memory_usage = Gauge('acousto_gen_gpu_memory_bytes',
                        'GPU memory usage in bytes', ['device'])

active_sessions = Gauge('acousto_gen_active_sessions',
                       'Number of active user sessions')

def monitor_optimization(method: str):
    """Decorator to monitor optimization operations."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                optimization_requests.labels(method=method, status='success').inc()
                return result
            except Exception as e:
                optimization_requests.labels(method=method, status='error').inc()
                raise
            finally:
                duration = time.time() - start_time
                optimization_duration.labels(method=method).observe(duration)
        return wrapper
    return decorator

class MetricsCollector:
    """Collect system metrics for monitoring."""
    
    def __init__(self):
        self.running = False
    
    def start_collection(self):
        """Start metrics collection."""
        self.running = True
        start_http_server(9090)
        
        while self.running:
            self.collect_gpu_metrics()
            self.collect_hardware_metrics()
            time.sleep(10)
    
    def collect_gpu_metrics(self):
        """Collect GPU utilization metrics."""
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device = f"cuda:{i}"
                    memory_allocated = torch.cuda.memory_allocated(i)
                    gpu_memory_usage.labels(device=device).set(memory_allocated)
        except Exception as e:
            logger.error(f"Failed to collect GPU metrics: {e}")
    
    def collect_hardware_metrics(self):
        """Collect hardware status metrics."""
        try:
            # Check hardware connections
            hardware_devices = get_connected_hardware()
            for device_name, status in hardware_devices.items():
                hardware_status.labels(device=device_name).set(1 if status else 0)
        except Exception as e:
            logger.error(f"Failed to collect hardware metrics: {e}")
```

### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "Acousto-Gen Monitoring",
    "panels": [
      {
        "title": "Optimization Requests Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(acousto_gen_optimization_requests_total[5m])",
            "legendFormat": "{{method}} - {{status}}"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "acousto_gen_gpu_memory_bytes",
            "legendFormat": "{{device}}"
          }
        ]
      },
      {
        "title": "Hardware Status",
        "type": "stat",
        "targets": [
          {
            "expr": "acousto_gen_hardware_status",
            "legendFormat": "{{device}}"
          }
        ]
      },
      {
        "title": "Optimization Duration",
        "type": "heatmap",
        "targets": [
          {
            "expr": "acousto_gen_optimization_duration_seconds_bucket",
            "legendFormat": "{{method}}"
          }
        ]
      }
    ]
  }
}
```

## Performance Optimization

### Load Balancing
```yaml
# load_balancer.yaml
apiVersion: v1
kind: Service
metadata:
  name: acousto-gen-loadbalancer
spec:
  selector:
    app: acousto-gen
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
  sessionAffinity: ClientIP  # Sticky sessions for WebSocket
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: acousto-gen-ingress
  annotations:
    nginx.ingress.kubernetes.io/load-balance: "ewma"
    nginx.ingress.kubernetes.io/upstream-hash-by: "$remote_addr"
spec:
  rules:
  - host: acousto-gen.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: acousto-gen-loadbalancer
            port:
              number: 80
```

### Caching Strategy
```python
# caching/manager.py
import redis
import pickle
from typing import Any, Optional
from functools import wraps

class CacheManager:
    """Manage optimization result caching."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_client = redis.from_url(redis_url)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        try:
            data = self.redis_client.get(key)
            return pickle.loads(data) if data else None
        except Exception:
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set cached value with TTL."""
        try:
            data = pickle.dumps(value)
            self.redis_client.setex(key, ttl, data)
        except Exception:
            pass
    
    def cache_optimization_result(self, ttl: int = 3600):
        """Decorator to cache optimization results."""
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                # Create cache key from function arguments
                cache_key = f"opt_{f.__name__}_{hash(str(args) + str(kwargs))}"
                
                # Try to get from cache
                result = self.get(cache_key)
                if result is not None:
                    return result
                
                # Compute and cache result
                result = f(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result
            return wrapper
        return decorator

# Usage
cache_manager = CacheManager()

@cache_manager.cache_optimization_result(ttl=7200)
def optimize_hologram(target_field, method="gradient", **kwargs):
    # Expensive optimization computation
    pass
```

## Backup and Recovery

### Automated Backup System
```bash
#!/bin/bash
# backup.sh - Automated backup system

set -euo pipefail

BACKUP_DIR="/opt/backups/acousto-gen"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="acousto-gen_backup_${TIMESTAMP}"

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# Backup application data
echo "Backing up application data..."
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}/data.tar.gz" \
    -C /opt/acousto-gen \
    data config logs

# Backup database
echo "Backing up database..."
pg_dump acousto_gen | gzip > "${BACKUP_DIR}/${BACKUP_NAME}/database.sql.gz"

# Backup Redis data
echo "Backing up Redis data..."
redis-cli --rdb "${BACKUP_DIR}/${BACKUP_NAME}/redis.rdb"

# Create manifest
cat > "${BACKUP_DIR}/${BACKUP_NAME}/manifest.json" << EOF
{
  "timestamp": "${TIMESTAMP}",
  "version": "$(acousto-gen --version)",
  "components": {
    "application_data": "data.tar.gz",
    "database": "database.sql.gz",
    "redis": "redis.rdb"
  },
  "size_mb": $(du -sm "${BACKUP_DIR}/${BACKUP_NAME}" | cut -f1)
}
EOF

# Compress entire backup
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" \
    -C "${BACKUP_DIR}" "${BACKUP_NAME}"

# Remove uncompressed backup
rm -rf "${BACKUP_DIR}/${BACKUP_NAME}"

# Clean old backups (keep last 30 days)
find "${BACKUP_DIR}" -name "acousto-gen_backup_*.tar.gz" \
    -mtime +30 -delete

echo "Backup completed: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
```

### Recovery Procedures
```bash
#!/bin/bash
# restore.sh - Recovery system

set -euo pipefail

BACKUP_FILE="$1"
RESTORE_DIR="/tmp/acousto-gen-restore"

if [[ ! -f "$BACKUP_FILE" ]]; then
    echo "Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Extract backup
mkdir -p "$RESTORE_DIR"
tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR"

BACKUP_NAME=$(basename "$BACKUP_FILE" .tar.gz)
BACKUP_PATH="${RESTORE_DIR}/${BACKUP_NAME}"

# Stop services
echo "Stopping Acousto-Gen services..."
systemctl stop acousto-gen
systemctl stop redis-server
systemctl stop postgresql

# Restore application data
echo "Restoring application data..."
tar -xzf "${BACKUP_PATH}/data.tar.gz" -C /opt/acousto-gen/

# Restore database
echo "Restoring database..."
systemctl start postgresql
sleep 5
dropdb --if-exists acousto_gen
createdb acousto_gen
gunzip -c "${BACKUP_PATH}/database.sql.gz" | psql acousto_gen

# Restore Redis data
echo "Restoring Redis data..."
cp "${BACKUP_PATH}/redis.rdb" /var/lib/redis/dump.rdb
chown redis:redis /var/lib/redis/dump.rdb

# Start services
echo "Starting services..."
systemctl start redis-server
systemctl start acousto-gen

# Verify restoration
sleep 10
if curl -f http://localhost:8000/health; then
    echo "✓ Restoration completed successfully"
else
    echo "✗ Restoration may have failed - check logs"
    exit 1
fi

# Cleanup
rm -rf "$RESTORE_DIR"
```

## Troubleshooting

### Common Issues and Solutions

#### High Memory Usage
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -20

# Check GPU memory
nvidia-smi

# Solutions:
# 1. Reduce batch size in configuration
# 2. Enable memory-efficient mode
# 3. Restart services to clear memory leaks
systemctl restart acousto-gen
```

#### Performance Degradation
```bash
# Check system load
htop
iostat -x 1

# Check GPU utilization
nvidia-smi -l 1

# Check network latency
ping -c 10 acousto-gen.example.com

# Solutions:
# 1. Scale horizontally (add more instances)
# 2. Optimize algorithm parameters
# 3. Check for resource contention
```

#### Hardware Connection Issues
```bash
# Check USB devices
lsusb | grep -i ultra

# Check permissions
ls -l /dev/ttyUSB*

# Check kernel messages
dmesg | tail -20

# Solutions:
# 1. Reconnect hardware
# 2. Update drivers
# 3. Check cable connections
# 4. Restart hardware interface service
```

### Log Analysis
```bash
# Monitor logs in real-time
tail -f /opt/acousto-gen/logs/acousto-gen.log

# Search for errors
grep -i error /opt/acousto-gen/logs/acousto-gen.log | tail -20

# Check performance metrics
grep "optimization_duration" /opt/acousto-gen/logs/acousto-gen.log

# Analyze patterns
awk '/ERROR/ {print $1, $2, $NF}' /opt/acousto-gen/logs/acousto-gen.log | sort | uniq -c
```

## Maintenance

### Regular Maintenance Tasks

#### Daily
- Monitor system health and performance metrics
- Check hardware connections and status
- Review error logs for anomalies
- Verify backup completion

#### Weekly
- Update system packages
- Rotate log files
- Check storage usage
- Test emergency procedures

#### Monthly
- Update Acousto-Gen to latest version
- Review and update configurations
- Audit security settings
- Performance optimization review

#### Quarterly
- Full system backup and restore testing
- Security vulnerability assessment
- Capacity planning review
- Hardware maintenance and calibration

### Maintenance Scripts
```bash
#!/bin/bash
# maintenance.sh - Regular maintenance tasks

# Update system packages
apt update && apt list --upgradable

# Check service status
systemctl is-active acousto-gen
systemctl is-active redis-server
systemctl is-active postgresql

# Check disk usage
df -h | grep -E "(/$|/opt|/var)"

# Check log file sizes
du -sh /opt/acousto-gen/logs/*

# Rotate logs if needed
if [[ $(du -m /opt/acousto-gen/logs/acousto-gen.log | cut -f1) -gt 100 ]]; then
    logrotate /etc/logrotate.d/acousto-gen
fi

# Test hardware connections
acousto-gen hardware-check

# Report status
echo "Maintenance check completed at $(date)"
```

This comprehensive deployment guide provides everything needed to deploy Acousto-Gen in production environments with proper security, monitoring, and maintenance procedures.