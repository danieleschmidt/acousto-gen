# Monitoring and Observability

This guide covers the comprehensive monitoring and observability setup for Acousto-Gen.

## Overview

Acousto-Gen implements a complete observability stack to ensure safe, reliable, and performant operation of acoustic holography systems.

### Key Components

- **Metrics Collection**: Prometheus for time-series data
- **Visualization**: Grafana dashboards for real-time monitoring
- **Alerting**: Alert rules for critical safety and performance issues
- **Logging**: Structured logging with correlation IDs
- **Tracing**: Distributed tracing for performance analysis
- **Health Monitoring**: Application and hardware health checks

## Quick Start

### Start Monitoring Stack

```bash
# Start full monitoring stack
docker-compose --profile monitoring up -d

# Access Grafana dashboard
open http://localhost:3000
# Username: admin, Password: acousto_admin
```

### View Metrics

```bash
# Access Prometheus
open http://localhost:9090

# View raw metrics
curl http://localhost:8080/metrics
```

## Metrics Categories

### Safety Metrics

Critical safety monitoring with immediate alerting:

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `acousto_gen_pressure_max` | Maximum acoustic pressure (Pa) | >5000 Pa |
| `acousto_gen_temperature_celsius` | Array temperature (°C) | >45°C |
| `acousto_gen_emergency_stop_active` | Emergency stop status | == 1 |
| `acousto_gen_intensity_max` | Maximum acoustic intensity (W/cm²) | >15 W/cm² |

### Hardware Metrics

Transducer array and system health:

```prometheus
# Array connection status
acousto_gen_array_connected{array_id="UL-001"}

# Element status
acousto_gen_failed_elements{array_id="UL-001"}

# Calibration timestamp
acousto_gen_calibration_timestamp{array_id="UL-001"}

# Power consumption
acousto_gen_power_watts{array_id="UL-001"}
```

### Performance Metrics

Application performance and resource usage:

```prometheus
# Optimization duration
acousto_gen_optimization_duration_seconds

# Memory usage  
process_resident_memory_bytes

# GPU utilization
acousto_gen_gpu_utilization_percent

# Request latency
acousto_gen_request_duration_seconds_bucket
```

### Application Metrics

Service health and operational metrics:

```prometheus
# Request rate
rate(acousto_gen_requests_total[5m])

# Error rate
rate(acousto_gen_errors_total[5m])

# Active optimizations
acousto_gen_active_optimizations

# Cache hit rate
acousto_gen_cache_hit_ratio
```

## Dashboards

### Main Dashboard: Acousto-Gen Overview

Access at: http://localhost:3000/d/acousto-gen-overview

**Panels:**
- Real-time safety gauges (pressure, temperature)
- Optimization performance trends
- Hardware status table
- Resource utilization graphs
- Recent alerts and annotations

### Safety Dashboard

Critical safety monitoring with large, clear indicators:

- **Emergency Status**: Large red/green indicator
- **Pressure Levels**: Multi-array pressure monitoring
- **Temperature Monitoring**: Thermal safety tracking
- **Safety System Health**: Monitor availability

### Performance Dashboard

Detailed performance analysis:

- **Optimization Metrics**: Duration, convergence, iterations
- **Resource Usage**: CPU, GPU, memory trends
- **Throughput Analysis**: Requests/second, batch processing
- **Efficiency Metrics**: Field quality, computation speed

### Hardware Dashboard

Comprehensive hardware monitoring:

- **Array Status**: Connection, calibration, health
- **Element Status**: Individual transducer health
- **Power Monitoring**: Consumption, efficiency
- **Environmental**: Temperature, humidity sensors

## Alerting

### Alert Severity Levels

| Level | Description | Response Time |
|-------|-------------|---------------|
| **Critical** | Safety risk, system failure | Immediate |
| **Warning** | Performance degradation | 15 minutes |
| **Info** | Operational notifications | 1 hour |

### Critical Safety Alerts

```yaml
# Immediate alerts (0s delay)
- AcousticPressureExceeded: >5000 Pa
- EmergencyStopTriggered: Emergency stop active
- SafetyMonitorDown: Safety system offline

# Fast alerts (5-10s delay)
- TemperatureHigh: >45°C array temperature
- IntensityExceeded: >15 W/cm² intensity
```

### Performance Alerts

```yaml
# Performance degradation
- OptimizationSlow: >30s optimization time
- MemoryUsageHigh: >8GB memory usage
- ErrorRateHigh: >0.1 errors/second
```

### Hardware Alerts

```yaml
# Hardware issues
- TransducerArrayDisconnected: Array not responding
- CalibrationExpired: >24h since calibration
- ElementFailure: Failed transducer elements
```

## Alert Configuration

### Notification Channels

Configure in `docker-compose.yml`:

```yaml
environment:
  - ALERTMANAGER_SLACK_WEBHOOK=your-webhook-url
  - ALERTMANAGER_EMAIL_SMTP=smtp.example.com
  - ALERTMANAGER_PAGERDUTY_KEY=your-key
```

### Alert Rules

Custom alert rules in `monitoring/prometheus/rules/`:

```yaml
# Custom alert example
- alert: CustomMetricHigh
  expr: your_custom_metric > threshold
  for: 60s
  labels:
    severity: warning
  annotations:
    summary: "Custom alert fired"
    description: "{{ $labels.instance }} metric is {{ $value }}"
```

## Health Checks

### Application Health

Built-in health check endpoints:

```bash
# Basic health
curl http://localhost:8080/health

# Detailed health with dependencies
curl http://localhost:8080/health/detailed

# Ready for traffic
curl http://localhost:8080/ready
```

### Hardware Health

Hardware-specific health monitoring:

```bash
# Array health check
curl http://localhost:8080/health/hardware

# Individual array status
curl http://localhost:8080/health/hardware/UL-001
```

### Safety System Health

Critical safety system monitoring:

```bash
# Safety system status
curl http://localhost:8080/health/safety

# Emergency stop test
curl -X POST http://localhost:8080/safety/test-emergency-stop
```

## Logging Configuration

### Structured Logging

JSON-formatted logs with correlation IDs:

```json
{
  "timestamp": "2025-01-01T12:00:00Z",
  "level": "INFO",
  "correlation_id": "req-12345",
  "component": "optimization",
  "message": "Optimization completed",
  "duration_ms": 1234,
  "iterations": 500,
  "convergence": 1e-6
}
```

### Log Levels

Configure log levels by component:

```bash
# Environment variables
ACOUSTO_LOG_LEVEL=INFO
ACOUSTO_LOG_LEVEL_OPTIMIZATION=DEBUG
ACOUSTO_LOG_LEVEL_SAFETY=DEBUG
ACOUSTO_LOG_LEVEL_HARDWARE=INFO
```

### Log Aggregation

Logs are collected and processed by:

- **Local Development**: Console output with colors
- **Docker**: Container logs via Docker logging driver
- **Production**: Loki for log aggregation and search

## Performance Monitoring

### Key Performance Indicators (KPIs)

Track these critical performance metrics:

```prometheus
# Optimization Performance
avg(acousto_gen_optimization_duration_seconds) by (method)

# System Throughput
rate(acousto_gen_requests_total[5m])

# Resource Efficiency
acousto_gen_gpu_utilization_percent / acousto_gen_power_watts

# Field Quality
acousto_gen_field_quality_score
```

### Performance Baselines

Establish performance baselines:

| Metric | Target | Baseline |
|--------|--------|----------|
| Optimization Time | <10s | 256 elements, 1000 iterations |
| Memory Usage | <4GB | Standard hologram generation |
| GPU Utilization | >80% | During optimization |
| Error Rate | <0.01% | Normal operation |

### Performance Regression Detection

Automated detection of performance regressions:

```prometheus
# Performance regression alert
- alert: OptimizationRegression
  expr: |
    (
      avg_over_time(acousto_gen_optimization_duration_seconds[1h]) /
      avg_over_time(acousto_gen_optimization_duration_seconds[24h] offset 7d)
    ) > 1.5
  for: 30m
  labels:
    severity: warning
  annotations:
    summary: "Performance regression detected"
    description: "Optimization 50% slower than last week"
```

## Custom Metrics

### Defining Custom Metrics

Add application-specific metrics:

```python
from prometheus_client import Counter, Histogram, Gauge

# Custom counters
levitation_attempts = Counter(
    'acousto_gen_levitation_attempts_total',
    'Total levitation attempts',
    ['particle_type', 'success']
)

# Custom histograms
field_calculation_time = Histogram(
    'acousto_gen_field_calculation_seconds',
    'Time spent calculating acoustic fields',
    ['resolution', 'method']
)

# Custom gauges
active_particles = Gauge(
    'acousto_gen_active_particles',
    'Number of actively levitated particles'
)
```

### Business Metrics

Track domain-specific metrics:

```prometheus
# Research metrics
acousto_gen_experiments_total
acousto_gen_papers_enabled_total
acousto_gen_research_groups_active

# Medical metrics  
acousto_gen_treatments_completed_total
acousto_gen_patient_safety_incidents_total
acousto_gen_thermal_dose_delivered

# Hardware metrics
acousto_gen_arrays_deployed_total
acousto_gen_calibration_drift_detected_total
acousto_gen_element_replacement_total
```

## Troubleshooting

### Common Issues

#### Metrics Not Appearing

```bash
# Check service health
curl http://localhost:8080/health

# Verify metrics endpoint
curl http://localhost:8080/metrics | grep acousto_gen

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

#### Alerts Not Firing

```bash
# Check alert rules
curl http://localhost:9090/api/v1/rules

# Verify alert manager
curl http://localhost:9093/api/v1/alerts

# Test alert expression
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=acousto_gen_pressure_max > 5000'
```

#### Dashboard Issues

```bash
# Check Grafana logs
docker-compose logs grafana

# Verify datasource connection
curl -u admin:acousto_admin \
  http://localhost:3000/api/datasources/proxy/1/api/v1/query?query=up
```

### Debug Mode

Enable debug monitoring:

```bash
# Start with debug logging
ACOUSTO_LOG_LEVEL=DEBUG docker-compose up

# Enable metrics debugging
ACOUSTO_METRICS_DEBUG=true docker-compose up
```

## Security Considerations

### Metrics Security

- **Access Control**: Secure Grafana and Prometheus access
- **Data Sensitivity**: Avoid exposing sensitive data in metrics
- **Network Security**: Use TLS for inter-service communication

### Alert Security

- **Notification Security**: Secure alert notification channels
- **False Positive Prevention**: Tune alert thresholds carefully
- **Alert Fatigue**: Implement alert routing and suppression

## Maintenance

### Regular Tasks

```bash
# Update dashboards
git pull && docker-compose restart grafana

# Clean old metrics data
docker exec prometheus sh -c 'find /prometheus -name "*.tmp" -delete'

# Backup Grafana settings
docker exec grafana sh -c 'tar czf /var/lib/grafana/backup.tar.gz /etc/grafana/'
```

### Capacity Planning

Monitor resource usage trends:

- **Storage Growth**: Prometheus data retention
- **Query Performance**: Dashboard load times
- **Alert Volume**: Alert frequency and patterns

For advanced monitoring scenarios, see:
- [Custom Metrics Guide](custom-metrics.md)
- [Alert Runbooks](../runbooks/)
- [Performance Tuning](performance-tuning.md)