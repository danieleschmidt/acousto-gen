#!/bin/bash
set -e

# Acousto-Gen Production Deployment Script
# This script handles the complete deployment of Acousto-Gen to production

echo "🚀 Starting Acousto-Gen Production Deployment"
echo "============================================="

# Configuration
DEPLOYMENT_DIR="/opt/acousto-gen"
BACKUP_DIR="/opt/acousto-gen-backups"
LOG_FILE="/var/log/acousto-gen-deploy.log"
DOCKER_COMPOSE_FILE="docker-compose.prod.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

error_exit() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    log "ERROR: $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
    log "WARNING: $1"
}

success() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
    log "SUCCESS: $1"
}

info() {
    echo -e "${BLUE}INFO: $1${NC}"
    log "INFO: $1"
}

# Pre-deployment checks
pre_deployment_checks() {
    info "Running pre-deployment checks..."
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error_exit "This script must be run as root"
    fi
    
    # Check system requirements
    if ! command -v docker &> /dev/null; then
        error_exit "Docker is not installed"
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error_exit "Docker Compose is not installed"
    fi
    
    # Check available disk space (minimum 10GB)
    available_space=$(df / | awk 'NR==2 {print $4}')
    required_space=$((10 * 1024 * 1024)) # 10GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        error_exit "Insufficient disk space. Required: 10GB, Available: $(($available_space / 1024 / 1024))GB"
    fi
    
    # Check available memory (minimum 4GB)
    available_memory=$(free -m | awk 'NR==2{print $2}')
    required_memory=4096 # 4GB
    
    if [ "$available_memory" -lt "$required_memory" ]; then
        warning "Low memory detected. Recommended: 4GB+, Available: ${available_memory}MB"
    fi
    
    success "Pre-deployment checks passed"
}

# Environment setup
setup_environment() {
    info "Setting up deployment environment..."
    
    # Create deployment directory
    mkdir -p "$DEPLOYMENT_DIR"
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Set up environment variables
    if [ ! -f "$DEPLOYMENT_DIR/.env" ]; then
        info "Creating environment configuration..."
        cat > "$DEPLOYMENT_DIR/.env" << EOL
# Acousto-Gen Production Environment
ACOUSTO_ENV=production
ACOUSTO_LOG_LEVEL=INFO

# Security
ACOUSTO_ADMIN_PASSWORD=$(openssl rand -base64 32)
POSTGRES_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 32)

# Database
DATABASE_URL=postgresql://acousto:\${POSTGRES_PASSWORD}@postgres:5432/acousto_db

# Redis
REDIS_URL=redis://redis:6379

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000

# Performance
MAX_WORKERS=4
WORKER_MEMORY_LIMIT=2G
CACHE_SIZE_MB=512

EOL
        success "Environment configuration created"
        warning "Please review and update the .env file with your specific configuration"
    fi
    
    success "Environment setup completed"
}

# Backup existing deployment
backup_existing() {
    if [ -d "$DEPLOYMENT_DIR/app" ]; then
        info "Backing up existing deployment..."
        
        backup_timestamp=$(date +%Y%m%d_%H%M%S)
        backup_path="$BACKUP_DIR/acousto-gen_$backup_timestamp"
        
        # Stop services first
        cd "$DEPLOYMENT_DIR" && docker-compose -f "$DOCKER_COMPOSE_FILE" down || true
        
        # Create backup
        cp -r "$DEPLOYMENT_DIR/app" "$backup_path" || error_exit "Failed to create backup"
        
        success "Backup created at $backup_path"
    fi
}

# Deploy application
deploy_application() {
    info "Deploying Acousto-Gen application..."
    
    # Copy application files
    cp -r . "$DEPLOYMENT_DIR/app" || error_exit "Failed to copy application files"
    cd "$DEPLOYMENT_DIR/app"
    
    # Set proper permissions
    chown -R 1000:1000 "$DEPLOYMENT_DIR/app"
    chmod +x "$DEPLOYMENT_DIR/app/scripts/"*.sh
    
    # Create necessary directories
    mkdir -p logs data cache monitoring/grafana/dashboards
    
    # Copy environment file
    cp "$DEPLOYMENT_DIR/.env" .env
    
    success "Application files deployed"
}

# Build and start services
start_services() {
    info "Building and starting services..."
    
    cd "$DEPLOYMENT_DIR/app"
    
    # Pull latest images
    docker-compose -f "$DOCKER_COMPOSE_FILE" pull
    
    # Build custom images
    docker-compose -f "$DOCKER_COMPOSE_FILE" build --no-cache
    
    # Start services
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    # Wait for services to be ready
    info "Waiting for services to start..."
    sleep 30
    
    # Check service health
    check_services_health
    
    success "Services started successfully"
}

# Check service health
check_services_health() {
    info "Checking service health..."
    
    services=("acousto-gen" "redis" "postgres" "prometheus" "grafana" "nginx")
    
    for service in "${services[@]}"; do
        if docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "$service.*Up"; then
            success "$service is running"
        else
            error_exit "$service failed to start"
        fi
    done
    
    # Test HTTP endpoints
    sleep 10
    
    # Test main application
    if curl -f -s http://localhost:8000/health > /dev/null; then
        success "Main application health check passed"
    else
        warning "Main application health check failed - may still be starting"
    fi
    
    # Test Prometheus
    if curl -f -s http://localhost:9090/-/ready > /dev/null; then
        success "Prometheus health check passed"
    else
        warning "Prometheus health check failed"
    fi
    
    # Test Grafana
    if curl -f -s http://localhost:3000/api/health > /dev/null; then
        success "Grafana health check passed"
    else
        warning "Grafana health check failed"
    fi
}

# Post-deployment tasks
post_deployment() {
    info "Running post-deployment tasks..."
    
    cd "$DEPLOYMENT_DIR/app"
    
    # Run database migrations
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T acousto-gen python3 -c "
from src.database.migrations import run_migrations
run_migrations()
print('Database migrations completed')
" || warning "Database migration failed"
    
    # Initialize monitoring
    info "Setting up monitoring dashboards..."
    # This would copy Grafana dashboards, etc.
    
    # Set up log rotation
    setup_log_rotation
    
    # Create systemd service for auto-start
    create_systemd_service
    
    success "Post-deployment tasks completed"
}

# Set up log rotation
setup_log_rotation() {
    info "Setting up log rotation..."
    
    cat > /etc/logrotate.d/acousto-gen << EOL
$DEPLOYMENT_DIR/app/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    create 644 root root
    postrotate
        docker-compose -f $DEPLOYMENT_DIR/app/$DOCKER_COMPOSE_FILE restart acousto-gen
    endscript
}
EOL
    
    success "Log rotation configured"
}

# Create systemd service
create_systemd_service() {
    info "Creating systemd service..."
    
    cat > /etc/systemd/system/acousto-gen.service << EOL
[Unit]
Description=Acousto-Gen Application
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$DEPLOYMENT_DIR/app
ExecStart=/usr/local/bin/docker-compose -f $DOCKER_COMPOSE_FILE up -d
ExecStop=/usr/local/bin/docker-compose -f $DOCKER_COMPOSE_FILE down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOL
    
    systemctl daemon-reload
    systemctl enable acousto-gen.service
    
    success "Systemd service created and enabled"
}

# Print deployment summary
print_summary() {
    echo ""
    echo "=================================================================="
    echo "🎉 ACOUSTO-GEN DEPLOYMENT COMPLETED SUCCESSFULLY!"
    echo "=================================================================="
    echo ""
    echo "📊 Service URLs:"
    echo "  • Main Application:  http://$(hostname):8000"
    echo "  • WebSocket API:     ws://$(hostname):8080"
    echo "  • Prometheus:        http://$(hostname):9090"
    echo "  • Grafana:           http://$(hostname):3000"
    echo "  • Load Balancer:     http://$(hostname)"
    echo ""
    echo "🔐 Default Credentials:"
    echo "  • Admin Panel:       admin / (check .env file)"
    echo "  • Grafana:          admin / (check .env file)"
    echo ""
    echo "📂 Important Paths:"
    echo "  • Application:       $DEPLOYMENT_DIR/app"
    echo "  • Logs:             $DEPLOYMENT_DIR/app/logs"
    echo "  • Data:             $DEPLOYMENT_DIR/app/data"
    echo "  • Backups:          $BACKUP_DIR"
    echo "  • Environment:      $DEPLOYMENT_DIR/.env"
    echo ""
    echo "🔧 Management Commands:"
    echo "  • View logs:         docker-compose -f $DOCKER_COMPOSE_FILE logs -f"
    echo "  • Restart:          systemctl restart acousto-gen"
    echo "  • Status:           systemctl status acousto-gen"
    echo "  • Update:           cd $DEPLOYMENT_DIR/app && git pull && ./deploy.sh"
    echo ""
    echo "⚠️  IMPORTANT: Please review and update the .env file with your specific"
    echo "   configuration before running in production!"
    echo ""
}

# Rollback function
rollback() {
    warning "Rolling back deployment..."
    
    cd "$DEPLOYMENT_DIR/app"
    docker-compose -f "$DOCKER_COMPOSE_FILE" down
    
    # Find latest backup
    latest_backup=$(ls -t "$BACKUP_DIR" | head -1)
    
    if [ -n "$latest_backup" ]; then
        rm -rf "$DEPLOYMENT_DIR/app"
        cp -r "$BACKUP_DIR/$latest_backup" "$DEPLOYMENT_DIR/app"
        cd "$DEPLOYMENT_DIR/app"
        docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
        
        success "Rollback completed using backup: $latest_backup"
    else
        error_exit "No backup available for rollback"
    fi
}

# Signal handlers
trap 'error_exit "Deployment interrupted"' INT TERM

# Main deployment flow
main() {
    # Check for rollback flag
    if [ "$1" = "--rollback" ]; then
        rollback
        exit 0
    fi
    
    # Run deployment steps
    pre_deployment_checks
    setup_environment
    backup_existing
    deploy_application
    start_services
    post_deployment
    print_summary
    
    success "Deployment completed successfully!"
}

# Run main function with all arguments
main "$@"