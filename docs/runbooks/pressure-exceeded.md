# Runbook: Acoustic Pressure Exceeded

## Alert: AcousticPressureExceeded

**Severity**: Critical  
**Alert Expression**: `acousto_gen_pressure_max > 5000`  
**For**: 0s (immediate)

## Description

The maximum acoustic pressure in the system has exceeded the safety limit of 5000 Pa. This is a critical safety issue that requires immediate attention to prevent potential harm.

## Immediate Actions (0-2 minutes)

### 1. Emergency Response
```bash
# Trigger emergency stop immediately
curl -X POST http://localhost:8080/safety/emergency-stop

# Verify all arrays are deactivated
curl http://localhost:8080/hardware/status | jq '.arrays[].active'
```

### 2. Verify Safety Systems
- Check that emergency stop has been activated
- Confirm all transducer arrays are powered down
- Verify no acoustic output is present

### 3. Clear the Area
- Ensure all personnel are at safe distance (>2m from arrays)
- Remove any objects from the acoustic field
- Post safety warnings if in shared facility

## Investigation (2-10 minutes)

### 4. Identify Root Cause

Check system logs for cause:
```bash
# Check recent pressure readings
curl http://localhost:9090/api/v1/query?query=acousto_gen_pressure_max[10m]

# Review optimization parameters
grep "pressure" /var/log/acousto-gen/optimization.log | tail -20

# Check for hardware malfunctions
curl http://localhost:8080/hardware/diagnostics
```

Common causes:
- **Optimization bug**: Incorrect phase calculations
- **Hardware malfunction**: Stuck transducer elements
- **Configuration error**: Wrong pressure limits set
- **Calibration drift**: Array calibration out of date
- **Software bug**: Safety check bypassed

### 5. Check System State
```bash
# Verify current pressure readings
curl http://localhost:8080/metrics | grep acousto_gen_pressure

# Check safety system status
curl http://localhost:8080/health/safety

# Review recent activities
tail -50 /var/log/acousto-gen/audit.log
```

## Resolution Steps (10-30 minutes)

### 6. Address Root Cause

**If Optimization Bug:**
```bash
# Reset optimization parameters to safe defaults
curl -X POST http://localhost:8080/config/reset-optimization

# Verify pressure limits are enforced
curl http://localhost:8080/config/safety-limits
```

**If Hardware Malfunction:**
```bash
# Run hardware diagnostics
curl -X POST http://localhost:8080/hardware/diagnose

# Check element status
curl http://localhost:8080/hardware/elements/status

# Disable failed elements if identified
curl -X POST http://localhost:8080/hardware/elements/disable \
  -d '{"elements": [123, 124]}'
```

**If Configuration Error:**
```bash
# Review and correct safety configuration
vim /etc/acousto-gen/safety.yaml

# Reload configuration
curl -X POST http://localhost:8080/config/reload
```

**If Calibration Issue:**
```bash
# Check calibration age
curl http://localhost:8080/hardware/calibration/status

# Trigger recalibration if needed
curl -X POST http://localhost:8080/hardware/calibration/start
```

### 7. Validate Fix
```bash
# Test with minimal power
curl -X POST http://localhost:8080/test/minimal-pressure

# Monitor pressure in real-time
watch -n 1 'curl -s http://localhost:8080/metrics | grep acousto_gen_pressure_max'

# Verify safety systems are active
curl http://localhost:8080/health/safety
```

## Safe Restart Procedure (30-45 minutes)

### 8. Pre-Start Checks
- [ ] Emergency stop system tested and functional
- [ ] All hardware diagnostics pass
- [ ] Safety limits properly configured
- [ ] Calibration current (<24 hours old)
- [ ] Monitoring systems operational
- [ ] Area clear of personnel and objects

### 9. Gradual Power-Up
```bash
# Start with 10% power limit
curl -X POST http://localhost:8080/config/power-limit -d '{"limit": 0.1}'

# Enable single array for testing
curl -X POST http://localhost:8080/hardware/enable -d '{"array_id": "test-array"}'

# Monitor pressure continuously
curl -X POST http://localhost:8080/monitoring/enable-continuous

# Gradually increase power if stable
# 10% → 25% → 50% → 75% → 100% (5 minutes each)
```

### 10. Verification
```bash
# Run full system test at low power
curl -X POST http://localhost:8080/test/system-check

# Verify all safety interlocks
curl -X POST http://localhost:8080/safety/test-all

# Check pressure remains within limits
curl http://localhost:9090/api/v1/query?query=max(acousto_gen_pressure_max[5m])
```

## Documentation and Follow-up

### 11. Incident Documentation
```bash
# Generate incident report
curl -X POST http://localhost:8080/incidents/generate-report \
  -d '{"type": "pressure_exceeded", "timestamp": "'$(date -Iseconds)'"}'

# Export relevant logs
docker exec acousto-gen-prod sh -c \
  'journalctl --since "30 minutes ago" > /tmp/incident-logs.txt'

# Capture system state
curl http://localhost:8080/system/state-dump > incident-state.json
```

Create incident report including:
- Timeline of events
- Root cause analysis
- Actions taken
- System changes made
- Lessons learned
- Prevention measures

### 12. Prevention Measures

**Immediate (same day):**
- Review and update safety limits
- Test emergency stop procedures
- Verify monitoring alert thresholds
- Update operator training materials

**Short-term (within week):**
- Implement additional safety checks
- Review optimization algorithms
- Enhance monitoring coverage
- Update maintenance procedures

**Long-term (within month):**
- Consider hardware redundancy
- Implement predictive monitoring
- Review safety protocols
- Conduct safety audit

## Alert Tuning

If this alert fires frequently due to false positives:

```yaml
# Adjust threshold (with extreme caution)
- alert: AcousticPressureExceeded
  expr: acousto_gen_pressure_max > 4800  # Lowered threshold
  for: 2s  # Small delay to avoid transient spikes
```

**CAUTION**: Only adjust thresholds after thorough safety review and approval from safety officer.

## Escalation

### Contact Information
- **Safety Officer**: safety@acousto-gen.org / +1-555-SAFETY
- **On-call Engineer**: oncall@acousto-gen.org / +1-555-ONCALL
- **Hardware Vendor**: support@ultraleap.com / +44-117-325-0270

### Escalation Triggers
- Cannot resolve within 45 minutes
- Repeated occurrences (>3 in 24 hours)
- Hardware damage suspected
- Personnel exposure possible
- Regulatory compliance concerns

## Related Runbooks
- [Temperature High](temperature-high.md)
- [Emergency Stop](emergency-stop.md)
- [Hardware Failure](hardware-failure.md)
- [Safety Monitor Down](safety-monitor-down.md)

## References
- Safety Limits Documentation: `/docs/safety/limits.md`
- Hardware Manual: `/docs/hardware/manual.pdf`
- Regulatory Guidelines: `/docs/compliance/fda-guidelines.md`
- Emergency Procedures: `/docs/emergency/procedures.md`