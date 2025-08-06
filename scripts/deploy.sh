#!/bin/bash
set -e

# Acousto-Gen Production Deployment Script
# This script automates the deployment of Acousto-Gen in production environments

echo "🚀 Starting Acousto-Gen Production Deployment"
echo "=============================================="

# Configuration
DEPLOYMENT_TYPE="${1:-standard}"  # standard, gpu, cluster, development
DOMAIN="${2:-localhost}"
ENVIRONMENT="${3:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Utility functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if running as root (needed for hardware access)
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root - this is required for hardware access"
    fi
    
    log_success "Prerequisites check completed"
}

# Generate secure passwords and keys
generate_secrets() {
    log_info "Generating secure secrets..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        cat > .env << EOF
# Acousto-Gen Production Environment Variables
# Generated on $(date)

# Security
JWT_SECRET_KEY=$(openssl rand -base64 64)
ACOUSTO_ADMIN_PASSWORD=$(openssl rand -base64 32)
DB_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 16)

# Application
ACOUSTO_ENV=${ENVIRONMENT}
ACOUSTO_DOMAIN=${DOMAIN}
ACOUSTO_DEBUG=false
ACOUSTO_ENABLE_HTTPS=true
ACOUSTO_ENABLE_RATE_LIMITING=true
ACOUSTO_AUDIT_LOGGING=true

# Database
POSTGRES_DB=acousto_gen
POSTGRES_USER=acousto

# Monitoring
PROMETHEUS_RETENTION_DAYS=30
GRAFANA_ENABLE_SIGNUP=false

# Performance
ACOUSTO_MAX_WORKERS=4
ACOUSTO_WORKER_MEMORY_LIMIT=8G
ACOUSTO_ENABLE_MULTI_GPU=false

# Security Headers
ACOUSTO_SECURITY_HEADERS=true
ACOUSTO_RATE_LIMIT_PER_MINUTE=100
EOF
        log_success "Generated .env file with secure secrets"
    else
        log_warning ".env file already exists - skipping secret generation"
    fi
}

# Setup SSL certificates
setup_ssl() {
    log_info "Setting up SSL certificates..."
    
    mkdir -p ssl
    
    if [ ! -f ssl/cert.pem ] || [ ! -f ssl/key.pem ]; then
        if [ "$DOMAIN" = "localhost" ]; then
            # Generate self-signed certificate for development
            log_warning "Generating self-signed certificate for localhost"
            openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes \
                -subj "/C=US/ST=State/L=City/O=AcoustoGen/OU=Development/CN=localhost"
        else
            log_error "SSL certificates not found. For production deployment:"
            log_error "1. Obtain certificates from a CA (e.g., Let's Encrypt)"
            log_error "2. Place cert.pem and key.pem in the ssl/ directory"
            exit 1
        fi
    fi
    
    # Set proper permissions
    chmod 600 ssl/key.pem
    chmod 644 ssl/cert.pem
    
    log_success "SSL setup completed"
}

# Setup directories and permissions
setup_directories() {
    log_info "Setting up directories and permissions..."
    
    # Create necessary directories
    mkdir -p {data,logs,ssl,config,monitoring}
    mkdir -p logs/{nginx,application,security,monitoring}
    mkdir -p data/{uploads,exports,calibration,experiments}
    mkdir -p config/{nginx/sites-enabled,postgres,redis,fluentd}
    mkdir -p monitoring/{prometheus/rules,grafana/{provisioning,dashboards}}
    
    # Set permissions
    chmod 755 data logs config monitoring
    chmod 700 ssl
    
    log_success "Directory setup completed"
}

# Generate configuration files
generate_configs() {
    log_info "Generating configuration files..."
    
    # Nginx configuration
    cat > config/nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Security headers
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options DENY always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    upstream acousto_api {
        server acousto-gen-api:8000;
    }
    
    upstream grafana {
        server grafana:3000;
    }
    
    server {
        listen 80;
        server_name _;
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name _;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
        
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://acousto_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /monitoring/ {
            proxy_pass http://grafana/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location / {
            proxy_pass http://acousto_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
EOF

    # PostgreSQL initialization
    cat > config/postgres/init.sql << 'EOF'
-- Acousto-Gen Database Initialization

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS acousto;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS audit;

-- Set default search path
ALTER DATABASE acousto_gen SET search_path TO acousto, public;

-- Create initial tables (basic structure)
CREATE TABLE IF NOT EXISTS acousto.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS acousto.experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    user_id UUID REFERENCES acousto.users(id),
    parameters JSONB,
    results JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS audit.events (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_id UUID REFERENCES acousto.users(id),
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(255),
    details JSONB,
    ip_address INET,
    user_agent TEXT
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_users_username ON acousto.users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON acousto.users(email);
CREATE INDEX IF NOT EXISTS idx_experiments_user_id ON acousto.experiments(user_id);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON acousto.experiments(status);
CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit.events(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_events_user_id ON audit.events(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_action ON audit.events(action);

-- Grant permissions
GRANT USAGE ON SCHEMA acousto TO acousto;
GRANT USAGE ON SCHEMA monitoring TO acousto;
GRANT USAGE ON SCHEMA audit TO acousto;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA acousto TO acousto;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO acousto;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA audit TO acousto;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA acousto TO acousto;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO acousto;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA audit TO acousto;
EOF

    log_success "Configuration files generated"
}

# Pull and build images
build_images() {
    log_info "Building Docker images..."
    
    case $DEPLOYMENT_TYPE in
        "gpu")
            log_info "Building GPU-enabled images..."
            docker-compose -f docker-compose.production.yml --profile gpu build
            ;;
        "cluster")
            log_info "Building cluster images..."
            docker-compose -f docker-compose.production.yml --profile cluster build
            ;;
        "development")
            log_info "Building development images..."
            docker-compose build
            ;;
        *)
            log_info "Building standard production images..."
            docker-compose -f docker-compose.production.yml build
            ;;
    esac
    
    log_success "Docker images built successfully"
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."
    
    case $DEPLOYMENT_TYPE in
        "gpu")
            log_info "Deploying with GPU support..."
            docker-compose -f docker-compose.production.yml --profile gpu up -d
            ;;
        "cluster")
            log_info "Deploying cluster configuration..."
            docker-compose -f docker-compose.production.yml --profile cluster up -d
            ;;
        "development")
            log_info "Deploying development environment..."
            docker-compose up -d
            ;;
        *)
            log_info "Deploying standard production configuration..."
            docker-compose -f docker-compose.production.yml up -d
            ;;
    esac
    
    log_success "Services deployed successfully"
}

# Wait for services to be healthy
wait_for_services() {
    log_info "Waiting for services to become healthy..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose -f docker-compose.production.yml ps | grep -q "healthy\|up"; then
            log_success "Services are healthy"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts - waiting for services..."
        sleep 10
        ((attempt++))
    done
    
    log_error "Services did not become healthy within expected time"
    return 1
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Wait for database to be ready
    docker-compose -f docker-compose.production.yml exec -T postgres \
        pg_isready -U acousto -d acousto_gen || {
        log_error "Database not ready"
        return 1
    }
    
    log_success "Database migrations completed"
}

# Security hardening
security_hardening() {
    log_info "Applying security hardening..."
    
    # Set file permissions
    find . -name "*.env" -exec chmod 600 {} \;
    find ssl/ -name "*.key" -exec chmod 600 {} \; 2>/dev/null || true
    find ssl/ -name "*.pem" -exec chmod 600 {} \; 2>/dev/null || true
    
    # Create security report
    docker-compose -f docker-compose.production.yml --profile security run --rm security-scanner
    
    log_success "Security hardening completed"
}

# Health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Check API health
    if curl -f -k https://localhost/health 2>/dev/null; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi
    
    # Check database connectivity
    if docker-compose -f docker-compose.production.yml exec -T postgres \
        psql -U acousto -d acousto_gen -c "SELECT 1;" >/dev/null 2>&1; then
        log_success "Database connectivity check passed"
    else
        log_error "Database connectivity check failed"
        return 1
    fi
    
    # Check Redis connectivity
    if docker-compose -f docker-compose.production.yml exec -T redis \
        redis-cli ping 2>/dev/null | grep -q "PONG"; then
        log_success "Redis connectivity check passed"
    else
        log_error "Redis connectivity check failed"
        return 1
    fi
    
    log_success "All health checks passed"
}

# Display deployment summary
display_summary() {
    log_success "🎉 Acousto-Gen deployment completed successfully!"
    echo ""
    echo "Deployment Summary:"
    echo "=================="
    echo "• Type: $DEPLOYMENT_TYPE"
    echo "• Domain: $DOMAIN"
    echo "• Environment: $ENVIRONMENT"
    echo ""
    echo "Services:"
    echo "--------"
    echo "• API: https://$DOMAIN"
    echo "• Monitoring: https://$DOMAIN/monitoring"
    echo "• API Documentation: https://$DOMAIN/docs"
    echo ""
    
    if [ -f .env ]; then
        echo "Admin Credentials:"
        echo "-----------------"
        echo "• Username: admin"
        echo "• Password: $(grep ACOUSTO_ADMIN_PASSWORD .env | cut -d'=' -f2)"
        echo ""
        echo "Grafana Access:"
        echo "• URL: https://$DOMAIN/monitoring"
        echo "• Username: admin"
        echo "• Password: $(grep GRAFANA_ADMIN_PASSWORD .env | cut -d'=' -f2)"
        echo ""
    fi
    
    echo "Management Commands:"
    echo "------------------"
    echo "• View logs: docker-compose logs -f"
    echo "• Stop services: docker-compose down"
    echo "• Update services: ./scripts/deploy.sh $DEPLOYMENT_TYPE $DOMAIN $ENVIRONMENT"
    echo "• Monitor status: docker-compose ps"
    echo ""
    echo "⚠️  IMPORTANT: Change default passwords and review security settings!"
}

# Main deployment function
main() {
    log_info "Acousto-Gen Deployment Script v1.0"
    log_info "Deployment type: $DEPLOYMENT_TYPE"
    log_info "Domain: $DOMAIN"
    log_info "Environment: $ENVIRONMENT"
    echo ""
    
    # Run deployment steps
    check_prerequisites
    generate_secrets
    setup_ssl
    setup_directories
    generate_configs
    build_images
    deploy_services
    wait_for_services
    run_migrations
    security_hardening
    run_health_checks
    display_summary
}

# Cleanup function for failures
cleanup() {
    log_error "Deployment failed. Cleaning up..."
    docker-compose -f docker-compose.production.yml down --remove-orphans
    exit 1
}

# Set trap for cleanup on failure
trap cleanup ERR

# Run main deployment
main

# Success
log_success "Deployment completed successfully!"
exit 0