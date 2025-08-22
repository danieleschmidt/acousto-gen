"""
Advanced Security Framework for Acousto-Gen Generation 2.
Implements authentication, authorization, input sanitization, and security monitoring.
"""

import hashlib
import secrets
import hmac
import time
import re
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
from functools import wraps
from threading import Lock
from collections import defaultdict
import jwt
import bcrypt

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    BASIC = "basic" 
    OPERATOR = "operator"
    ADMINISTRATOR = "administrator"
    DEVELOPER = "developer"


class ThreatType(Enum):
    """Types of security threats."""
    INJECTION = "injection"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_EXPOSURE = "data_exposure"
    DOS_ATTACK = "dos_attack"
    MALICIOUS_INPUT = "malicious_input"
    PRIVILEGE_ESCALATION = "privilege_escalation"


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: float
    event_type: ThreatType
    severity: str
    source_ip: Optional[str]
    user_id: Optional[str]
    details: str
    blocked: bool
    context: Dict[str, Any]


@dataclass
class UserCredentials:
    """User credential information."""
    user_id: str
    username: str
    password_hash: str
    security_level: SecurityLevel
    created_at: float
    last_login: Optional[float] = None
    failed_attempts: int = 0
    locked_until: Optional[float] = None
    api_key_hash: Optional[str] = None


class InputSanitizer:
    """Sanitizes and validates all user inputs."""
    
    # Dangerous patterns that should be blocked
    DANGEROUS_PATTERNS = [
        # SQL injection patterns
        r"(union\s+select|drop\s+table|delete\s+from|insert\s+into)",
        # Command injection patterns  
        r"([;&|`$]|\.\.\/|\/etc\/|\/bin\/|cmd\.exe)",
        # Script injection patterns
        r"(<script|javascript:|vbscript:|on\w+\s*=)",
        # Path traversal
        r"(\.\.\/|\.\.\\|\.\.\%2f|\.\.\%5c)",
        # Protocol handlers
        r"(file:|ftp:|http:|https:|data:|mailto:)",
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.DANGEROUS_PATTERNS]
    
    def sanitize_string(self, value: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            raise ValueError("Input must be a string")
        
        # Length check
        if len(value) > max_length:
            raise ValueError(f"Input too long: {len(value)} > {max_length}")
        
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(value):
                raise ValueError(f"Potentially dangerous input pattern detected")
        
        # Basic sanitization
        sanitized = value.strip()
        
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')
        
        return sanitized
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe filesystem operations."""
        if not filename:
            raise ValueError("Filename cannot be empty")
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)
        
        # Remove leading/trailing spaces and dots
        filename = filename.strip(' .')
        
        # Check for reserved names (Windows)
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'{name}{i}' for name in ['COM', 'LPT'] for i in range(1, 10)]
        if filename.upper() in reserved_names:
            filename = f"safe_{filename}"
        
        # Ensure reasonable length
        if len(filename) > 255:
            filename = filename[:255]
        
        return filename
    
    def validate_numeric_range(self, value: Union[int, float], min_val: float, max_val: float, param_name: str) -> Union[int, float]:
        """Validate numeric parameter is within safe range."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{param_name} must be numeric")
        
        if not (min_val <= value <= max_val):
            raise ValueError(f"{param_name} must be between {min_val} and {max_val}, got {value}")
        
        return value
    
    def validate_acoustic_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate acoustic parameters for safety."""
        safe_params = {}
        
        if "frequency" in params:
            safe_params["frequency"] = self.validate_numeric_range(
                params["frequency"], 1000, 200000, "frequency"
            )
        
        if "pressure" in params:
            safe_params["pressure"] = self.validate_numeric_range(
                params["pressure"], 0, 5000, "pressure"  # 5 kPa max
            )
        
        if "iterations" in params:
            safe_params["iterations"] = self.validate_numeric_range(
                params["iterations"], 1, 10000, "iterations"
            )
        
        if "learning_rate" in params:
            safe_params["learning_rate"] = self.validate_numeric_range(
                params["learning_rate"], 0.00001, 1.0, "learning_rate"
            )
        
        # Validate array inputs
        if "phases" in params:
            phases = params["phases"]
            if not isinstance(phases, (list, tuple)):
                raise ValueError("Phases must be a list or tuple")
            if len(phases) > 1000:  # Reasonable limit
                raise ValueError(f"Too many phase elements: {len(phases)} > 1000")
            safe_params["phases"] = phases
        
        if "positions" in params:
            positions = params["positions"]
            if not isinstance(positions, (list, tuple)):
                raise ValueError("Positions must be a list or tuple")
            for i, pos in enumerate(positions):
                if not isinstance(pos, (list, tuple)) or len(pos) != 3:
                    raise ValueError(f"Position {i} must be 3D coordinate [x, y, z]")
                # Validate position range (reasonable workspace)
                x, y, z = pos
                if not (-0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.1 <= z <= 1.0):
                    raise ValueError(f"Position {i} outside safe workspace")
            safe_params["positions"] = positions
        
        return safe_params


class AuthenticationManager:
    """Manages user authentication and session security."""
    
    def __init__(self, secret_key: Optional[str] = None, token_expiry: int = 3600):
        """Initialize authentication manager."""
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.token_expiry = token_expiry  # seconds
        self.users = {}  # In production, this would be a database
        self.active_sessions = {}
        self.failed_attempts = defaultdict(int)
        self.lockout_time = 900  # 15 minutes
        self._lock = Lock()
    
    def hash_password(self, password: str) -> str:
        """Hash password securely."""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        except Exception:
            return False
    
    def create_user(self, username: str, password: str, security_level: SecurityLevel) -> str:
        """Create new user account."""
        with self._lock:
            if username in self.users:
                raise ValueError("Username already exists")
            
            user_id = secrets.token_urlsafe(16)
            password_hash = self.hash_password(password)
            api_key = secrets.token_urlsafe(32)
            
            self.users[username] = UserCredentials(
                user_id=user_id,
                username=username,
                password_hash=password_hash,
                security_level=security_level,
                created_at=time.time(),
                api_key_hash=self.hash_password(api_key)
            )
            
            logger.info(f"Created user {username} with security level {security_level.value}")
            return api_key
    
    def authenticate(self, username: str, password: str, source_ip: Optional[str] = None) -> Optional[str]:
        """Authenticate user and return JWT token."""
        with self._lock:
            # Check if account is locked
            lockout_key = f"{username}:{source_ip or 'unknown'}"
            if lockout_key in self.failed_attempts:
                if self.failed_attempts[lockout_key] >= 5:  # Max 5 attempts
                    logger.warning(f"Account {username} locked due to failed attempts")
                    return None
            
            # Verify user exists
            if username not in self.users:
                self.failed_attempts[lockout_key] += 1
                return None
            
            user = self.users[username]
            
            # Check if user account is locked
            if user.locked_until and time.time() < user.locked_until:
                return None
            
            # Verify password
            if not self.verify_password(password, user.password_hash):
                self.failed_attempts[lockout_key] += 1
                return None
            
            # Reset failed attempts on successful login
            if lockout_key in self.failed_attempts:
                del self.failed_attempts[lockout_key]
            
            # Update last login
            user.last_login = time.time()
            
            # Create JWT token
            payload = {
                "user_id": user.user_id,
                "username": username,
                "security_level": user.security_level.value,
                "exp": time.time() + self.token_expiry,
                "iat": time.time()
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm='HS256')
            
            # Store active session
            self.active_sessions[token] = {
                "user_id": user.user_id,
                "username": username,
                "created_at": time.time(),
                "last_activity": time.time()
            }
            
            logger.info(f"User {username} authenticated successfully")
            return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return user info."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check if session is still active
            if token in self.active_sessions:
                session = self.active_sessions[token]
                session["last_activity"] = time.time()
                return payload
            
            return None
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def revoke_token(self, token: str):
        """Revoke authentication token."""
        if token in self.active_sessions:
            del self.active_sessions[token]
            logger.info("Token revoked")


class AuthorizationManager:
    """Manages access control and permissions."""
    
    # Permission matrix: operation -> required security level
    PERMISSIONS = {
        "read_system_status": SecurityLevel.PUBLIC,
        "read_field_data": SecurityLevel.BASIC,
        "optimize_hologram": SecurityLevel.OPERATOR,
        "control_hardware": SecurityLevel.OPERATOR,
        "modify_safety_limits": SecurityLevel.ADMINISTRATOR,
        "create_user": SecurityLevel.ADMINISTRATOR,
        "system_configuration": SecurityLevel.ADMINISTRATOR,
        "debug_access": SecurityLevel.DEVELOPER,
        "raw_hardware_access": SecurityLevel.DEVELOPER,
    }
    
    @staticmethod
    def check_permission(user_security_level: SecurityLevel, operation: str) -> bool:
        """Check if user has permission for operation."""
        if operation not in AuthorizationManager.PERMISSIONS:
            logger.warning(f"Unknown operation: {operation}")
            return False
        
        required_level = AuthorizationManager.PERMISSIONS[operation]
        
        # Security level hierarchy
        level_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.BASIC: 1,
            SecurityLevel.OPERATOR: 2,
            SecurityLevel.ADMINISTRATOR: 3,
            SecurityLevel.DEVELOPER: 4
        }
        
        user_level_value = level_hierarchy.get(user_security_level, -1)
        required_level_value = level_hierarchy.get(required_level, 999)
        
        return user_level_value >= required_level_value


class SecurityMonitor:
    """Monitors and detects security threats."""
    
    def __init__(self):
        self.events = []
        self.threat_counters = defaultdict(int)
        self._lock = Lock()
        
        # Rate limiting
        self.rate_limits = {
            "api_calls": {"limit": 100, "window": 60},  # 100 calls per minute
            "authentication": {"limit": 10, "window": 300},  # 10 attempts per 5 min
            "optimization": {"limit": 50, "window": 3600},  # 50 optimizations per hour
        }
        self.rate_tracking = defaultdict(list)
    
    def log_security_event(self, event_type: ThreatType, details: str, 
                          source_ip: Optional[str] = None, user_id: Optional[str] = None,
                          severity: str = "medium", blocked: bool = False, **context):
        """Log a security event."""
        with self._lock:
            event = SecurityEvent(
                timestamp=time.time(),
                event_type=event_type,
                severity=severity,
                source_ip=source_ip,
                user_id=user_id,
                details=details,
                blocked=blocked,
                context=context
            )
            
            self.events.append(event)
            self.threat_counters[event_type] += 1
            
            # Keep only recent events (last 1000)
            if len(self.events) > 1000:
                self.events = self.events[-1000:]
            
            # Log based on severity
            log_level = {
                "low": logging.INFO,
                "medium": logging.WARNING, 
                "high": logging.ERROR,
                "critical": logging.CRITICAL
            }.get(severity, logging.WARNING)
            
            logger.log(log_level, f"Security event: {event_type.value} - {details}")
    
    def check_rate_limit(self, operation: str, identifier: str) -> bool:
        """Check if operation exceeds rate limit."""
        if operation not in self.rate_limits:
            return True
        
        config = self.rate_limits[operation]
        current_time = time.time()
        window_start = current_time - config["window"]
        
        # Clean old entries
        key = f"{operation}:{identifier}"
        self.rate_tracking[key] = [
            timestamp for timestamp in self.rate_tracking[key] 
            if timestamp > window_start
        ]
        
        # Check limit
        if len(self.rate_tracking[key]) >= config["limit"]:
            self.log_security_event(
                ThreatType.DOS_ATTACK,
                f"Rate limit exceeded for {operation}: {len(self.rate_tracking[key])} > {config['limit']}",
                source_ip=identifier if "." in identifier else None,
                severity="high",
                blocked=True
            )
            return False
        
        # Record this request
        self.rate_tracking[key].append(current_time)
        return True
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect security anomalies in recent events."""
        anomalies = []
        
        # Check for repeated failed authentication
        recent_events = [e for e in self.events if time.time() - e.timestamp < 3600]
        auth_failures = [e for e in recent_events if e.event_type == ThreatType.AUTHENTICATION and not e.blocked]
        
        if len(auth_failures) > 10:
            anomalies.append({
                "type": "repeated_auth_failures",
                "count": len(auth_failures),
                "severity": "high"
            })
        
        # Check for unusual activity patterns
        user_activity = defaultdict(int)
        for event in recent_events:
            if event.user_id:
                user_activity[event.user_id] += 1
        
        for user_id, count in user_activity.items():
            if count > 200:  # More than 200 events per hour
                anomalies.append({
                    "type": "unusual_activity",
                    "user_id": user_id,
                    "count": count,
                    "severity": "medium"
                })
        
        return anomalies


class SecurityFramework:
    """Main security framework orchestrator."""
    
    def __init__(self, enable_authentication: bool = True):
        """Initialize security framework."""
        self.sanitizer = InputSanitizer()
        self.auth_manager = AuthenticationManager() if enable_authentication else None
        self.auth_enabled = enable_authentication
        self.monitor = SecurityMonitor()
        
        # Create default admin user if authentication is enabled
        if self.auth_enabled and self.auth_manager:
            try:
                admin_key = self.auth_manager.create_user(
                    "admin", "change_me_please", SecurityLevel.ADMINISTRATOR
                )
                logger.info(f"Default admin user created. API Key: {admin_key}")
            except ValueError:
                pass  # User already exists
    
    def secure_endpoint(self, operation: str, required_level: SecurityLevel = SecurityLevel.BASIC):
        """Decorator for securing API endpoints."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Extract request context
                request_info = kwargs.get('request_info', {})
                source_ip = request_info.get('source_ip', 'unknown')
                auth_token = request_info.get('auth_token')
                
                # Check rate limiting
                if not self.monitor.check_rate_limit("api_calls", source_ip):
                    raise PermissionError("Rate limit exceeded")
                
                # Authenticate if enabled
                user_info = None
                if self.auth_enabled and self.auth_manager:
                    if not auth_token:
                        self.monitor.log_security_event(
                            ThreatType.AUTHENTICATION,
                            f"Missing authentication token for {operation}",
                            source_ip=source_ip,
                            severity="medium"
                        )
                        raise PermissionError("Authentication required")
                    
                    user_info = self.auth_manager.verify_token(auth_token)
                    if not user_info:
                        self.monitor.log_security_event(
                            ThreatType.AUTHENTICATION,
                            f"Invalid authentication token for {operation}",
                            source_ip=source_ip,
                            severity="high"
                        )
                        raise PermissionError("Invalid authentication")
                    
                    # Check authorization
                    user_level = SecurityLevel(user_info["security_level"])
                    if not AuthorizationManager.check_permission(user_level, operation):
                        self.monitor.log_security_event(
                            ThreatType.AUTHORIZATION,
                            f"Insufficient permissions for {operation}",
                            source_ip=source_ip,
                            user_id=user_info.get("user_id"),
                            severity="high"
                        )
                        raise PermissionError("Insufficient permissions")
                
                # Sanitize inputs
                sanitized_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        sanitized_kwargs[key] = self.sanitizer.sanitize_string(value)
                    elif isinstance(value, dict) and key == 'parameters':
                        sanitized_kwargs[key] = self.sanitizer.validate_acoustic_parameters(value)
                    else:
                        sanitized_kwargs[key] = value
                
                # Execute function
                try:
                    result = func(*args, **sanitized_kwargs)
                    
                    # Log successful operation
                    self.monitor.log_security_event(
                        ThreatType.AUTHORIZATION,
                        f"Successful {operation}",
                        source_ip=source_ip,
                        user_id=user_info.get("user_id") if user_info else None,
                        severity="low"
                    )
                    
                    return result
                    
                except Exception as e:
                    # Log security-relevant exceptions
                    self.monitor.log_security_event(
                        ThreatType.MALICIOUS_INPUT,
                        f"Exception in {operation}: {str(e)[:200]}",
                        source_ip=source_ip,
                        user_id=user_info.get("user_id") if user_info else None,
                        severity="medium"
                    )
                    raise
            
            return wrapper
        return decorator
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        anomalies = self.monitor.detect_anomalies()
        
        return {
            "authentication_enabled": self.auth_enabled,
            "active_sessions": len(self.auth_manager.active_sessions) if self.auth_manager else 0,
            "recent_events": len([e for e in self.monitor.events if time.time() - e.timestamp < 3600]),
            "threat_counts": dict(self.monitor.threat_counters),
            "anomalies": anomalies,
            "security_level": "high" if self.auth_enabled else "basic"
        }


# Global security framework instance
security_framework = SecurityFramework(enable_authentication=True)