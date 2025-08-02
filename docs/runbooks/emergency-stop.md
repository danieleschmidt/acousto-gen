# Runbook: Emergency Stop Activated

## Alert: EmergencyStopTriggered

**Severity**: Critical  
**Alert Expression**: `acousto_gen_emergency_stop_active == 1`  
**For**: 0s (immediate)

## Description

The emergency stop system has been activated, immediately halting all acoustic output. This could be triggered manually, automatically by safety systems, or due to system fault detection.

## Immediate Response (0-5 minutes)

### 1. Acknowledge and Assess
- **Confirm emergency stop status is active**
- **Verify all acoustic output has ceased**
- **Check for immediate safety hazards**
- **Account for all personnel in the area**

```bash
# Verify emergency stop status
curl http://localhost:8080/safety/emergency-stop/status

# Check all arrays are deactivated
curl http://localhost:8080/hardware/arrays/status | jq '.[] | select(.active == true)'

# Review immediate safety status
curl http://localhost:8080/safety/status
```

### 2. Safety Assessment
- [ ] Area is safe and secure
- [ ] No personnel in danger
- [ ] No equipment damage visible
- [ ] Fire/smoke/unusual sounds absent
- [ ] Emergency services not required

### 3. Initial Logging
```bash
# Capture emergency stop event details
curl http://localhost:8080/events/emergency-stop/details > emergency-stop-$(date +%Y%m%d_%H%M%S).json

# Get system state at time of stop
curl http://localhost:8080/system/state-snapshot > system-state-$(date +%Y%m%d_%H%M%S).json
```

## Investigation Phase (5-20 minutes)

### 4. Determine Trigger Source

Check what triggered the emergency stop:

```bash
# Check emergency stop trigger log
grep "emergency_stop" /var/log/acousto-gen/safety.log | tail -10

# Review safety system alerts
curl http://localhost:9090/api/v1/query?query=acousto_gen_safety_alerts[10m]

# Check manual activation logs
curl http://localhost:8080/audit/manual-stops?since=10m
```

**Common Triggers:**

| Trigger Type | Investigation Steps |
|--------------|-------------------|
| **Manual Activation** | Check physical button logs, user authentication |
| **Pressure Limit** | Review pressure readings, optimization parameters |
| **Temperature Alert** | Check thermal sensors, cooling system status |
| **Hardware Fault** | Run diagnostics, check element status |
| **Software Error** | Review error logs, exception traces |
| **External Signal** | Check interlock systems, external safety devices |

### 5. Assess System State

```bash
# Check current system health
curl http://localhost:8080/health/detailed

# Review hardware diagnostics
curl -X POST http://localhost:8080/hardware/diagnose/all

# Check safety system integrity
curl http://localhost:8080/safety/self-test
```

### 6. Review Recent Activity

```bash
# Check last 15 minutes of operations
tail -n 100 /var/log/acousto-gen/operations.log

# Review recent optimizations
curl http://localhost:8080/history/optimizations?limit=10

# Check configuration changes
curl http://localhost:8080/audit/config-changes?since=1h
```

## Resolution Process (20-60 minutes)

### 7. Address Root Cause

**For Pressure/Safety Limit Triggers:**
```bash
# Review and verify safety limits
curl http://localhost:8080/config/safety-limits

# Check calibration status
curl http://localhost:8080/hardware/calibration/status

# Verify optimization parameters are safe
curl http://localhost:8080/config/optimization
```

**For Hardware Fault Triggers:**
```bash
# Run comprehensive hardware diagnostics
curl -X POST http://localhost:8080/hardware/diagnose/comprehensive

# Check element health
curl http://localhost:8080/hardware/elements/health

# Test individual arrays
for array in $(curl -s http://localhost:8080/hardware/arrays | jq -r '.[].id'); do
  curl -X POST "http://localhost:8080/hardware/test/array/$array"
done
```

**For Software Error Triggers:**
```bash
# Check application logs for errors
grep -i error /var/log/acousto-gen/application.log | tail -20

# Review exception stack traces
curl http://localhost:8080/errors/recent

# Check system resources
curl http://localhost:8080/system/resources
```

**For External Interlock Triggers:**
```bash
# Check interlock system status
curl http://localhost:8080/interlocks/status

# Test interlock connections
curl -X POST http://localhost:8080/interlocks/test

# Review interlock configuration
curl http://localhost:8080/config/interlocks
```

### 8. Verify Fixes

After addressing the root cause:

```bash
# Run safety system validation
curl -X POST http://localhost:8080/safety/validate

# Test emergency stop functionality
curl -X POST http://localhost:8080/safety/test-emergency-stop

# Verify all safety interlocks
curl -X POST http://localhost:8080/safety/test-interlocks
```

## Reset Procedure (60-90 minutes)

### 9. Pre-Reset Checks

Before resetting the emergency stop:

- [ ] Root cause identified and resolved
- [ ] All safety systems tested and functional
- [ ] Hardware diagnostics pass
- [ ] Area is clear and safe
- [ ] Appropriate personnel present
- [ ] Monitoring systems active

```bash
# Final safety validation
curl -X POST http://localhost:8080/safety/pre-reset-check

# Verify monitoring is active
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health == "up")'
```

### 10. Reset Emergency Stop

**Physical Reset (if manual button pressed):**
1. Rotate emergency stop button clockwise to release
2. Verify button indicator light changes to green
3. Confirm system recognizes reset

**Software Reset:**
```bash
# Reset emergency stop state
curl -X POST http://localhost:8080/safety/emergency-stop/reset

# Verify reset successful
curl http://localhost:8080/safety/emergency-stop/status
```

### 11. Graduated Restart

**Phase 1: System Initialization**
```bash
# Initialize safety systems
curl -X POST http://localhost:8080/safety/initialize

# Start monitoring
curl -X POST http://localhost:8080/monitoring/start

# Initialize hardware interfaces
curl -X POST http://localhost:8080/hardware/initialize
```

**Phase 2: Hardware Preparation**
```bash
# Enable hardware interfaces (no power)
curl -X POST http://localhost:8080/hardware/enable-interfaces

# Run pre-power diagnostics
curl -X POST http://localhost:8080/hardware/diagnose/pre-power

# Load calibration data
curl -X POST http://localhost:8080/hardware/load-calibration
```

**Phase 3: Low Power Test**
```bash
# Set power limit to 10%
curl -X POST http://localhost:8080/config/power-limit -d '{"limit": 0.1}'

# Enable single test array
curl -X POST http://localhost:8080/hardware/enable -d '{"array_id": "test-array"}'

# Run low power test
curl -X POST http://localhost:8080/test/low-power
```

**Phase 4: Gradual Power Increase**
```bash
# Increase power gradually: 10% → 25% → 50% → 75% → 100%
for power in 0.25 0.5 0.75 1.0; do
  curl -X POST http://localhost:8080/config/power-limit -d "{\"limit\": $power}"
  sleep 300  # Wait 5 minutes between increases
  curl http://localhost:8080/safety/status  # Check safety status
done
```

**Phase 5: Full System Enable**
```bash
# Enable all arrays
curl -X POST http://localhost:8080/hardware/enable-all

# Run full system test
curl -X POST http://localhost:8080/test/full-system

# Resume normal operations
curl -X POST http://localhost:8080/operations/resume
```

## Post-Incident Activities

### 12. Documentation

```bash
# Generate incident report
curl -X POST http://localhost:8080/incidents/create \
  -d '{
    "type": "emergency_stop",
    "trigger": "determined_cause",
    "duration": "total_downtime_minutes",
    "impact": "brief_description"
  }'

# Export detailed logs
docker exec acousto-gen-prod sh -c \
  'journalctl --since "2 hours ago" --until "now" > /tmp/emergency-stop-logs.txt'

# Capture system configuration
curl http://localhost:8080/config/export > post-incident-config.json
```

**Include in incident report:**
- Timeline of events
- Root cause analysis
- Actions taken
- System changes made
- Personnel involved
- Downtime duration
- Impact assessment
- Lessons learned
- Prevention recommendations

### 13. Regulatory Reporting

If required by local regulations:
- Notify relevant authorities
- File safety incident reports
- Update safety documentation
- Schedule safety audit if required

### 14. Follow-up Actions

**Immediate (within 24 hours):**
- [ ] Brief all operators on incident
- [ ] Update emergency procedures if needed
- [ ] Test emergency stop system
- [ ] Review monitoring alert thresholds

**Short-term (within 1 week):**
- [ ] Implement additional safety measures
- [ ] Update training materials
- [ ] Review maintenance schedules
- [ ] Conduct team debriefing

**Long-term (within 1 month):**
- [ ] Safety system audit
- [ ] Process improvement review
- [ ] Consider system upgrades
- [ ] Update emergency response procedures

## Prevention Strategies

### Proactive Monitoring
```bash
# Set up predictive alerts
curl -X POST http://localhost:8080/alerts/create \
  -d '{
    "name": "pressure_trending_high",
    "condition": "acousto_gen_pressure_max > 4000",
    "action": "warn"
  }'

# Regular safety system tests
curl -X POST http://localhost:8080/maintenance/schedule \
  -d '{
    "task": "emergency_stop_test",
    "frequency": "weekly"
  }'
```

### Training and Procedures
- Regular emergency response drills
- Updated safety training
- Clear escalation procedures
- Operator competency verification

## Related Documentation

- [Pressure Exceeded Runbook](pressure-exceeded.md)
- [Temperature High Runbook](temperature-high.md) 
- [Hardware Failure Runbook](hardware-failure.md)
- [Safety System Manual](../safety/system-manual.md)
- [Emergency Procedures](../emergency/procedures.md)

## Contacts

### Emergency Contacts
- **Site Safety Officer**: +1-555-SAFETY-1
- **Facility Manager**: +1-555-FACILITY
- **Emergency Services**: 911 (if life safety concern)

### Technical Contacts  
- **On-call Engineer**: +1-555-ONCALL
- **Hardware Vendor**: +44-117-325-0270
- **Safety System Vendor**: +1-555-SAFETY-2

### Management Contacts
- **Department Head**: head@organization.org
- **Safety Manager**: safety@organization.org
- **Operations Manager**: ops@organization.org