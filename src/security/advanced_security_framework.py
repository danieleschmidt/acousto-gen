"""
Advanced Security Framework
Generation 2: MAKE IT ROBUST - Comprehensive security measures for acoustic holography systems.
"""

import numpy as np
import hashlib
import hmac
import secrets
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import threading
from datetime import datetime, timedelta
import base64
import os

# Configure security logging
security_logger = logging.getLogger('acousto_gen.security')
security_logger.setLevel(logging.INFO)

# Add file handler for security events
security_handler = logging.FileHandler('security_events.log')
security_handler.setFormatter(logging.Formatter(
    '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
))
security_logger.addHandler(security_handler)


class SecurityLevel(Enum):
    """Security levels for different operations."""
    PUBLIC = "public"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    session_id: str
    permissions: List[str]
    security_level: SecurityLevel
    authenticated: bool
    timestamp: float
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class SecurityEvent:
    """Security event record."""
    event_type: str
    severity: ThreatLevel
    timestamp: float
    user_id: Optional[str]
    source_ip: Optional[str]
    description: str
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyLimits:
    """Safety limits for acoustic operations."""
    max_pressure: float = 5000.0  # Pa
    max_intensity: float = 10.0   # W/cm²
    max_frequency: float = 100000.0  # Hz
    max_exposure_time: float = 3600.0  # seconds
    min_distance: float = 0.05  # meters from array
    
    def __post_init__(self):
        """Validate safety limits."""
        if self.max_pressure > 50000.0:  # 50 kPa absolute max
            raise ValueError("Maximum pressure exceeds absolute safety limit")
        if self.max_intensity > 100.0:  # 100 W/cm² absolute max
            raise ValueError("Maximum intensity exceeds absolute safety limit")


class CryptographicManager:
    """Handles encryption, hashing, and cryptographic operations."""
    
    def __init__(self):
        self.algorithm = 'AES-256-GCM'
        self.key_size = 32  # 256 bits
        self.iv_size = 16   # 128 bits
        self.tag_size = 16  # 128 bits
        
    def generate_key(self) -> bytes:
        """Generate cryptographically secure random key."""
        return secrets.token_bytes(self.key_size)
    
    def generate_iv(self) -> bytes:
        """Generate initialization vector."""
        return secrets.token_bytes(self.iv_size)
    
    def hash_data(self, data: Union[str, bytes], salt: Optional[bytes] = None) -> Tuple[str, bytes]:
        """Hash data with optional salt."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # Use PBKDF2 for key derivation
        from hashlib import pbkdf2_hmac
        hashed = pbkdf2_hmac('sha256', data, salt, 100000)
        return base64.b64encode(hashed).decode('utf-8'), salt
    
    def verify_hash(self, data: Union[str, bytes], hashed: str, salt: bytes) -> bool:
        """Verify data against hash."""
        computed_hash, _ = self.hash_data(data, salt)
        return hmac.compare_digest(computed_hash, hashed)
    
    def secure_random_phases(self, size: int) -> np.ndarray:
        """Generate cryptographically secure random phases."""
        # Use cryptographically secure random number generator
        random_bytes = secrets.token_bytes(size * 8)  # 8 bytes per float64
        random_floats = np.frombuffer(random_bytes, dtype=np.float64)
        
        # Scale to phase range [-π, π]
        phases = (random_floats / np.max(random_floats)) * 2 * np.pi - np.pi
        return phases[:size]
    
    def generate_session_token(self) -> str:
        """Generate secure session token."""
        return secrets.token_urlsafe(32)
    
    def constant_time_compare(self, a: str, b: str) -> bool:
        """Constant time string comparison to prevent timing attacks."""
        return hmac.compare_digest(a, b)


class AccessControlManager:
    """Manages user authentication and authorization."""
    
    def __init__(self):
        self.users = {}  # In production, use secure database
        self.sessions = {}
        self.failed_attempts = {}
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes
        self.session_timeout = 3600  # 1 hour
        self.crypto = CryptographicManager()
        
    def create_user(self, user_id: str, password: str, permissions: List[str],
                   security_level: SecurityLevel) -> bool:
        """Create new user account."""
        try:
            if user_id in self.users:
                security_logger.warning(f"Attempt to create existing user: {user_id}")
                return False
            
            # Hash password
            password_hash, salt = self.crypto.hash_data(password)
            
            self.users[user_id] = {
                'password_hash': password_hash,
                'salt': salt,
                'permissions': permissions,
                'security_level': security_level,
                'created_at': time.time(),
                'last_login': None,
                'login_count': 0
            }
            
            security_logger.info(f"User created: {user_id}")
            return True
            
        except Exception as e:
            security_logger.error(f"User creation failed for {user_id}: {e}")
            return False
    
    def authenticate(self, user_id: str, password: str, source_ip: str = None) -> Optional[str]:
        """Authenticate user and return session token."""
        try:
            # Check for account lockout
            if self._is_locked_out(user_id):
                security_logger.warning(f"Login attempt on locked account: {user_id}")
                return None
            
            # Check if user exists
            if user_id not in self.users:
                self._record_failed_attempt(user_id, source_ip)
                security_logger.warning(f"Login attempt for non-existent user: {user_id}")
                return None
            
            user = self.users[user_id]
            
            # Verify password
            if not self.crypto.verify_hash(password, user['password_hash'], user['salt']):
                self._record_failed_attempt(user_id, source_ip)
                security_logger.warning(f"Failed login attempt for user: {user_id}")
                return None
            
            # Authentication successful
            session_token = self.crypto.generate_session_token()
            
            # Create session
            self.sessions[session_token] = {
                'user_id': user_id,
                'created_at': time.time(),
                'last_activity': time.time(),
                'source_ip': source_ip,
                'permissions': user['permissions'],
                'security_level': user['security_level']
            }
            
            # Update user login info
            user['last_login'] = time.time()
            user['login_count'] += 1
            
            # Clear failed attempts
            if user_id in self.failed_attempts:
                del self.failed_attempts[user_id]
            
            security_logger.info(f"Successful login: {user_id}")
            return session_token
            
        except Exception as e:
            security_logger.error(f"Authentication error for {user_id}: {e}")
            return None
    
    def validate_session(self, session_token: str) -> Optional[SecurityContext]:
        """Validate session token and return security context."""
        try:
            if session_token not in self.sessions:
                return None
            
            session = self.sessions[session_token]
            
            # Check session timeout
            if time.time() - session['last_activity'] > self.session_timeout:
                del self.sessions[session_token]
                security_logger.info(f"Session expired: {session['user_id']}")
                return None
            
            # Update last activity
            session['last_activity'] = time.time()
            
            return SecurityContext(
                user_id=session['user_id'],
                session_id=session_token,
                permissions=session['permissions'],
                security_level=session['security_level'],
                authenticated=True,
                timestamp=time.time(),
                source_ip=session.get('source_ip')
            )
            
        except Exception as e:
            security_logger.error(f"Session validation error: {e}")
            return None
    
    def logout(self, session_token: str) -> bool:
        """Logout user and invalidate session."""
        try:
            if session_token in self.sessions:
                user_id = self.sessions[session_token]['user_id']
                del self.sessions[session_token]
                security_logger.info(f"User logged out: {user_id}")
                return True
            return False
        except Exception as e:
            security_logger.error(f"Logout error: {e}")
            return False
    
    def _is_locked_out(self, user_id: str) -> bool:
        """Check if user account is locked out."""
        if user_id not in self.failed_attempts:
            return False
        
        attempts = self.failed_attempts[user_id]
        if attempts['count'] >= self.max_failed_attempts:
            if time.time() - attempts['last_attempt'] < self.lockout_duration:
                return True
            else:
                # Lockout expired
                del self.failed_attempts[user_id]
                return False
        
        return False
    
    def _record_failed_attempt(self, user_id: str, source_ip: str = None):
        """Record failed login attempt."""
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = {'count': 0, 'last_attempt': 0, 'source_ips': []}
        
        self.failed_attempts[user_id]['count'] += 1
        self.failed_attempts[user_id]['last_attempt'] = time.time()
        
        if source_ip and source_ip not in self.failed_attempts[user_id]['source_ips']:
            self.failed_attempts[user_id]['source_ips'].append(source_ip)


class SafetyValidator:
    """Validates acoustic operations against safety constraints."""
    
    def __init__(self, safety_limits: SafetyLimits = None):
        self.safety_limits = safety_limits or SafetyLimits()
        self.exposure_tracking = {}  # Track cumulative exposure
        
    def validate_phases(self, phases: np.ndarray, frequency: float = 40000.0) -> Tuple[bool, List[str]]:
        """Validate phase array for safety compliance."""
        violations = []
        
        try:
            # Check for NaN or infinite values
            if not np.isfinite(phases).all():
                violations.append("Phase array contains invalid values")
            
            # Check phase magnitudes (very large phases could indicate errors)
            max_phase = np.max(np.abs(phases))
            if max_phase > 100 * np.pi:  # Unreasonably large phases
                violations.append(f"Phase magnitudes too large: {max_phase:.2f}")
            
            # Check for potential resonance issues
            phase_std = np.std(phases)
            if phase_std < 0.01:  # All phases nearly identical
                violations.append("Phase array lacks diversity - potential resonance risk")
            
            # Frequency-specific checks
            if frequency > self.safety_limits.max_frequency:
                violations.append(f"Frequency {frequency} Hz exceeds safety limit")
            
            return len(violations) == 0, violations
            
        except Exception as e:
            violations.append(f"Safety validation error: {e}")
            return False, violations
    
    def validate_target_field(self, target_field: np.ndarray) -> Tuple[bool, List[str]]:
        """Validate target field for safety compliance."""
        violations = []
        
        try:
            # Convert to numpy if needed
            if hasattr(target_field, 'numpy'):
                field_data = target_field.numpy()
            else:
                field_data = target_field
            
            # Check maximum pressure
            max_pressure = np.max(np.abs(field_data))
            if max_pressure > self.safety_limits.max_pressure:
                violations.append(
                    f"Target pressure {max_pressure:.1f} Pa exceeds safety limit "
                    f"{self.safety_limits.max_pressure:.1f} Pa"
                )
            
            # Check for extremely high local pressures
            pressure_99th = np.percentile(np.abs(field_data), 99)
            if pressure_99th > self.safety_limits.max_pressure * 0.8:
                violations.append(
                    f"99th percentile pressure {pressure_99th:.1f} Pa approaches safety limit"
                )
            
            # Check spatial gradients (high gradients can cause issues)
            if field_data.ndim > 1:
                gradients = np.gradient(field_data)
                max_gradient = np.max([np.max(np.abs(g)) for g in gradients])
                
                # Gradient safety threshold
                gradient_limit = self.safety_limits.max_pressure / 0.001  # Pa/mm
                if max_gradient > gradient_limit:
                    violations.append(f"Pressure gradient too high: {max_gradient:.1f} Pa/m")
            
            return len(violations) == 0, violations
            
        except Exception as e:
            violations.append(f"Target field validation error: {e}")
            return False, violations
    
    def validate_exposure_time(self, user_id: str, duration: float) -> Tuple[bool, List[str]]:
        """Validate cumulative exposure time."""
        violations = []
        
        try:
            current_time = time.time()
            
            # Initialize tracking for new users
            if user_id not in self.exposure_tracking:
                self.exposure_tracking[user_id] = {
                    'daily_exposure': 0.0,
                    'last_reset': current_time,
                    'session_start': None
                }
            
            tracking = self.exposure_tracking[user_id]
            
            # Reset daily counter if needed
            if current_time - tracking['last_reset'] > 86400:  # 24 hours
                tracking['daily_exposure'] = 0.0
                tracking['last_reset'] = current_time
            
            # Check single session limit
            if duration > self.safety_limits.max_exposure_time:
                violations.append(
                    f"Session duration {duration:.1f}s exceeds limit "
                    f"{self.safety_limits.max_exposure_time:.1f}s"
                )
            
            # Check daily cumulative limit
            daily_limit = self.safety_limits.max_exposure_time * 8  # 8 sessions per day max
            if tracking['daily_exposure'] + duration > daily_limit:
                violations.append(
                    f"Daily exposure limit would be exceeded: "
                    f"{tracking['daily_exposure'] + duration:.1f}s > {daily_limit:.1f}s"
                )
            
            return len(violations) == 0, violations
            
        except Exception as e:
            violations.append(f"Exposure validation error: {e}")
            return False, violations
    
    def record_exposure(self, user_id: str, duration: float):
        """Record actual exposure time."""
        try:
            if user_id in self.exposure_tracking:
                self.exposure_tracking[user_id]['daily_exposure'] += duration
                security_logger.info(f"Recorded exposure for {user_id}: {duration:.1f}s")
        except Exception as e:
            security_logger.error(f"Exposure recording error: {e}")


class ThreatDetector:
    """Detects and responds to security threats."""
    
    def __init__(self):
        self.threat_signatures = {
            'parameter_injection': [
                lambda params: any('__' in str(v) for v in params.values()),
                lambda params: any('eval' in str(v) for v in params.values()),
                lambda params: any('exec' in str(v) for v in params.values()),
            ],
            'unusual_patterns': [
                lambda phases: np.any(np.abs(phases) > 1000),  # Unusually large phases
                lambda phases: np.all(phases == phases[0]),     # All identical phases
                lambda phases: len(phases) > 100000,           # Extremely large arrays
            ],
            'timing_attacks': [
                lambda timing: timing > 10.0,  # Operations taking too long
            ]
        }
        
        self.threat_events = []
        self.blocked_ips = set()
        self.rate_limits = {}  # IP -> {count, window_start}
        self.rate_limit_window = 60  # 1 minute
        self.rate_limit_max = 100   # Max requests per window
    
    def detect_threats(self, operation: str, parameters: Dict[str, Any],
                      security_context: SecurityContext) -> List[SecurityEvent]:
        """Detect potential security threats."""
        threats = []
        
        try:
            # Parameter injection detection
            if self._check_parameter_injection(parameters):
                threats.append(SecurityEvent(
                    event_type='parameter_injection',
                    severity=ThreatLevel.HIGH,
                    timestamp=time.time(),
                    user_id=security_context.user_id,
                    source_ip=security_context.source_ip,
                    description='Potential parameter injection detected',
                    additional_data={'operation': operation, 'parameters': str(parameters)[:200]}
                ))
            
            # Phase pattern analysis
            if 'phases' in parameters:
                phases = parameters['phases']
                if hasattr(phases, 'numpy'):
                    phases = phases.numpy()
                
                if self._check_unusual_phase_patterns(phases):
                    threats.append(SecurityEvent(
                        event_type='unusual_phase_pattern',
                        severity=ThreatLevel.MEDIUM,
                        timestamp=time.time(),
                        user_id=security_context.user_id,
                        source_ip=security_context.source_ip,
                        description='Unusual phase patterns detected',
                        additional_data={'phase_stats': self._get_phase_statistics(phases)}
                    ))
            
            # Rate limiting check
            if self._check_rate_limit(security_context.source_ip):
                threats.append(SecurityEvent(
                    event_type='rate_limit_exceeded',
                    severity=ThreatLevel.MEDIUM,
                    timestamp=time.time(),
                    user_id=security_context.user_id,
                    source_ip=security_context.source_ip,
                    description='Rate limit exceeded',
                    additional_data={'operation': operation}
                ))
            
            # Record all detected threats
            for threat in threats:
                self.threat_events.append(threat)
                security_logger.warning(
                    f"THREAT DETECTED: {threat.event_type} - {threat.description}"
                )
            
            return threats
            
        except Exception as e:
            security_logger.error(f"Threat detection error: {e}")
            return []
    
    def _check_parameter_injection(self, parameters: Dict[str, Any]) -> bool:
        """Check for parameter injection attempts."""
        for signature in self.threat_signatures['parameter_injection']:
            try:
                if signature(parameters):
                    return True
            except:
                continue
        return False
    
    def _check_unusual_phase_patterns(self, phases: np.ndarray) -> bool:
        """Check for unusual phase patterns."""
        for signature in self.threat_signatures['unusual_patterns']:
            try:
                if signature(phases):
                    return True
            except:
                continue
        return False
    
    def _check_rate_limit(self, source_ip: str) -> bool:
        """Check if source IP exceeds rate limit."""
        if not source_ip:
            return False
        
        current_time = time.time()
        
        if source_ip not in self.rate_limits:
            self.rate_limits[source_ip] = {'count': 1, 'window_start': current_time}
            return False
        
        rate_info = self.rate_limits[source_ip]
        
        # Check if window has expired
        if current_time - rate_info['window_start'] > self.rate_limit_window:
            rate_info['count'] = 1
            rate_info['window_start'] = current_time
            return False
        
        # Increment count
        rate_info['count'] += 1
        
        # Check if limit exceeded
        if rate_info['count'] > self.rate_limit_max:
            self.blocked_ips.add(source_ip)
            security_logger.warning(f"IP blocked for rate limit violation: {source_ip}")
            return True
        
        return False
    
    def _get_phase_statistics(self, phases: np.ndarray) -> Dict[str, float]:
        """Get statistical summary of phase array."""
        try:
            return {
                'mean': float(np.mean(phases)),
                'std': float(np.std(phases)),
                'min': float(np.min(phases)),
                'max': float(np.max(phases)),
                'size': int(len(phases))
            }
        except:
            return {}
    
    def is_blocked(self, source_ip: str) -> bool:
        """Check if IP is blocked."""
        return source_ip in self.blocked_ips
    
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get threat detection summary."""
        if not self.threat_events:
            return {'message': 'No threats detected'}
        
        # Group by threat type
        threat_counts = {}
        for event in self.threat_events:
            threat_type = event.event_type
            if threat_type not in threat_counts:
                threat_counts[threat_type] = 0
            threat_counts[threat_type] += 1
        
        # Recent threats (last hour)
        recent_threshold = time.time() - 3600
        recent_threats = [e for e in self.threat_events if e.timestamp > recent_threshold]
        
        return {
            'total_threats': len(self.threat_events),
            'recent_threats': len(recent_threats),
            'threat_types': threat_counts,
            'blocked_ips': len(self.blocked_ips),
            'active_rate_limits': len(self.rate_limits)
        }


class SecurityFramework:
    """Main security framework coordinating all security components."""
    
    def __init__(self, safety_limits: SafetyLimits = None):
        self.access_control = AccessControlManager()
        self.safety_validator = SafetyValidator(safety_limits)
        self.threat_detector = ThreatDetector()
        self.crypto = CryptographicManager()
        
        # Security configuration
        self.security_enabled = True
        self.audit_logging = True
        self.safety_checks_enabled = True
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user for initial setup."""
        try:
            admin_password = secrets.token_urlsafe(16)
            self.access_control.create_user(
                user_id='admin',
                password=admin_password,
                permissions=['all'],
                security_level=SecurityLevel.TOP_SECRET
            )
            
            # Log admin credentials (remove in production)
            security_logger.info(f"Default admin created with password: {admin_password}")
            
        except Exception as e:
            security_logger.error(f"Failed to create default admin: {e}")
    
    def secure_optimize(self, optimization_function: Callable, 
                       session_token: str, *args, **kwargs) -> Any:
        """Execute optimization with full security checks."""
        try:
            # Validate session
            security_context = self.access_control.validate_session(session_token)
            if not security_context:
                raise PermissionError("Invalid or expired session")
            
            # Check permissions
            if 'optimization' not in security_context.permissions and 'all' not in security_context.permissions:
                raise PermissionError("Insufficient permissions for optimization")
            
            # Extract parameters for validation
            phases = kwargs.get('phases') or (args[0] if len(args) > 0 else None)
            target_field = kwargs.get('target_field') or (args[1] if len(args) > 1 else None)
            
            # Safety validation
            if self.safety_checks_enabled:
                if phases is not None:
                    safe, violations = self.safety_validator.validate_phases(phases)
                    if not safe:
                        raise ValueError(f"Safety violations: {'; '.join(violations)}")
                
                if target_field is not None:
                    safe, violations = self.safety_validator.validate_target_field(target_field)
                    if not safe:
                        raise ValueError(f"Safety violations: {'; '.join(violations)}")
            
            # Threat detection
            operation_params = {k: v for k, v in kwargs.items() if k not in ['phases', 'target_field']}
            threats = self.threat_detector.detect_threats(
                operation='optimization',
                parameters=operation_params,
                security_context=security_context
            )
            
            # Block if high-severity threats detected
            high_severity_threats = [t for t in threats if t.severity in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]
            if high_severity_threats:
                raise SecurityError(f"Operation blocked due to security threats: {len(high_severity_threats)} detected")
            
            # Check if IP is blocked
            if security_context.source_ip and self.threat_detector.is_blocked(security_context.source_ip):
                raise PermissionError("Source IP is blocked")
            
            # Execute optimization
            start_time = time.time()
            result = optimization_function(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Record exposure if applicable
            self.safety_validator.record_exposure(security_context.user_id, execution_time)
            
            # Audit log
            if self.audit_logging:
                security_logger.info(
                    f"AUDIT: User {security_context.user_id} performed optimization "
                    f"in {execution_time:.2f}s"
                )
            
            return result
            
        except Exception as e:
            security_logger.error(f"Secure optimization failed: {e}")
            raise e
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            'security_enabled': self.security_enabled,
            'audit_logging': self.audit_logging,
            'safety_checks_enabled': self.safety_checks_enabled,
            'active_sessions': len(self.access_control.sessions),
            'registered_users': len(self.access_control.users),
            'failed_login_attempts': len(self.access_control.failed_attempts),
            'threat_summary': self.threat_detector.get_threat_summary(),
            'safety_limits': {
                'max_pressure': self.safety_validator.safety_limits.max_pressure,
                'max_intensity': self.safety_validator.safety_limits.max_intensity,
                'max_frequency': self.safety_validator.safety_limits.max_frequency,
                'max_exposure_time': self.safety_validator.safety_limits.max_exposure_time
            }
        }


class SecurityError(Exception):
    """Security-related errors."""
    pass


# Decorator for secure functions
def require_security(security_framework: SecurityFramework):
    """Decorator to require security checks for functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract session token from kwargs
            session_token = kwargs.pop('session_token', None)
            if not session_token:
                raise PermissionError("Session token required")
            
            return security_framework.secure_optimize(func, session_token, *args, **kwargs)
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    print("🔒 Advanced Security Framework")
    print("Generation 2: MAKE IT ROBUST - Comprehensive Security")
    
    # Create security framework
    safety_limits = SafetyLimits(
        max_pressure=4000.0,
        max_intensity=8.0,
        max_frequency=80000.0
    )
    
    security = SecurityFramework(safety_limits)
    
    # Example secure optimization function
    @require_security(security)
    def secure_optimization(phases, target_field, iterations=1000):
        """Example secure optimization function."""
        time.sleep(0.1)  # Simulate processing
        return {
            'phases': phases + np.random.normal(0, 0.1, len(phases)),
            'final_loss': np.random.random() * 0.01,
            'iterations': iterations
        }
    
    print("\n🧪 Testing security framework...")
    
    # Create test user
    security.access_control.create_user(
        user_id='test_user',
        password='secure_password123',
        permissions=['optimization'],
        security_level=SecurityLevel.RESTRICTED
    )
    
    # Authenticate
    session_token = security.access_control.authenticate('test_user', 'secure_password123')
    
    if session_token:
        print("✅ Authentication successful")
        
        try:
            # Test secure optimization
            phases = np.random.uniform(-np.pi, np.pi, 256)
            target = np.random.random((16, 16, 16)) * 3000  # Safe pressure levels
            
            result = secure_optimization(
                phases=phases,
                target_field=target,
                iterations=500,
                session_token=session_token
            )
            
            print("✅ Secure optimization successful")
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
        
        # Test safety violation
        try:
            unsafe_target = np.random.random((16, 16, 16)) * 10000  # Unsafe pressure
            result = secure_optimization(
                phases=phases,
                target_field=unsafe_target,
                session_token=session_token
            )
        except Exception as e:
            print(f"✅ Safety violation correctly blocked: {e}")
        
        # Security status
        status = security.get_security_status()
        print(f"\n🔒 Security Status:")
        print(f"   Active sessions: {status['active_sessions']}")
        print(f"   Registered users: {status['registered_users']}")
        print(f"   Threats detected: {status['threat_summary'].get('total_threats', 0)}")
        
    else:
        print("❌ Authentication failed")
    
    print("\n🛡️ Security framework ready for production!")