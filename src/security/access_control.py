"""
Access control and authentication system for acoustic holography applications.
Provides role-based access control, API authentication, and audit logging.
"""

import hashlib
import secrets
import time
import jwt
import bcrypt
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import threading
import logging


class UserRole(Enum):
    """User roles with different privilege levels."""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    OPERATOR = "operator"
    OBSERVER = "observer"
    GUEST = "guest"


class Permission(Enum):
    """System permissions."""
    # Hardware control
    HARDWARE_CONTROL = "hardware_control"
    HARDWARE_CONFIG = "hardware_config"
    EMERGENCY_STOP = "emergency_stop"
    
    # Field generation
    FIELD_GENERATE = "field_generate"
    FIELD_MODIFY = "field_modify"
    FIELD_VIEW = "field_view"
    
    # Safety controls
    SAFETY_OVERRIDE = "safety_override"
    SAFETY_CONFIG = "safety_config"
    SAFETY_VIEW = "safety_view"
    
    # Data access
    DATA_EXPORT = "data_export"
    DATA_DELETE = "data_delete"
    DATA_VIEW = "data_view"
    
    # System administration
    USER_MANAGE = "user_manage"
    SYSTEM_CONFIG = "system_config"
    AUDIT_VIEW = "audit_view"
    
    # Research functions
    RESEARCH_MODE = "research_mode"
    ALGORITHM_MODIFY = "algorithm_modify"
    CALIBRATION_MODIFY = "calibration_modify"


@dataclass
class User:
    """User account information."""
    username: str
    email: str
    role: UserRole
    permissions: Set[Permission]
    created_at: float
    last_login: float = 0
    failed_attempts: int = 0
    locked_until: float = 0
    session_token: Optional[str] = None
    api_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions
    
    def is_locked(self) -> bool:
        """Check if account is locked."""
        return time.time() < self.locked_until


@dataclass
class AuditEvent:
    """Audit log event."""
    timestamp: float
    user: str
    action: str
    resource: str
    result: str  # success, failure, denied
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AccessControl:
    """
    Role-based access control system for acoustic holography.
    
    Provides user authentication, authorization, session management,
    and comprehensive audit logging.
    """
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        session_timeout: int = 3600,  # 1 hour
        max_failed_attempts: int = 5,
        lockout_duration: int = 900,  # 15 minutes
        audit_log_path: str = "audit.log"
    ):
        """
        Initialize access control system.
        
        Args:
            secret_key: JWT secret key (generated if not provided)
            session_timeout: Session timeout in seconds
            max_failed_attempts: Max failed login attempts before lockout
            lockout_duration: Account lockout duration in seconds
            audit_log_path: Path to audit log file
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.session_timeout = session_timeout
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration = lockout_duration
        
        # User storage (in production, use proper database)
        self.users: Dict[str, User] = {}
        self.password_hashes: Dict[str, bytes] = {}
        
        # Active sessions
        self.active_sessions: Dict[str, str] = {}  # token -> username
        self.session_lock = threading.Lock()
        
        # API keys
        self.api_keys: Dict[str, str] = {}  # key -> username
        
        # Role permissions mapping
        self.role_permissions = self._setup_role_permissions()
        
        # Audit logging
        self.audit_logger = self._setup_audit_logging(audit_log_path)
        self.audit_events: List[AuditEvent] = []
        
        # Security settings
        self.require_strong_passwords = True
        self.enable_2fa = False  # Two-factor authentication (placeholder)
        
        # Rate limiting
        self.rate_limits: Dict[str, List[float]] = {}  # ip -> request_times
        self.rate_limit_window = 300  # 5 minutes
        self.rate_limit_max_requests = 100
        
        # Create default admin user
        self._create_default_admin()
    
    def _setup_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Setup default role permissions."""
        return {
            UserRole.ADMIN: {
                Permission.HARDWARE_CONTROL,
                Permission.HARDWARE_CONFIG,
                Permission.EMERGENCY_STOP,
                Permission.FIELD_GENERATE,
                Permission.FIELD_MODIFY,
                Permission.FIELD_VIEW,
                Permission.SAFETY_OVERRIDE,
                Permission.SAFETY_CONFIG,
                Permission.SAFETY_VIEW,
                Permission.DATA_EXPORT,
                Permission.DATA_DELETE,
                Permission.DATA_VIEW,
                Permission.USER_MANAGE,
                Permission.SYSTEM_CONFIG,
                Permission.AUDIT_VIEW,
                Permission.RESEARCH_MODE,
                Permission.ALGORITHM_MODIFY,
                Permission.CALIBRATION_MODIFY,
            },
            UserRole.RESEARCHER: {
                Permission.HARDWARE_CONTROL,
                Permission.EMERGENCY_STOP,
                Permission.FIELD_GENERATE,
                Permission.FIELD_MODIFY,
                Permission.FIELD_VIEW,
                Permission.SAFETY_VIEW,
                Permission.DATA_EXPORT,
                Permission.DATA_VIEW,
                Permission.RESEARCH_MODE,
                Permission.ALGORITHM_MODIFY,
            },
            UserRole.OPERATOR: {
                Permission.HARDWARE_CONTROL,
                Permission.EMERGENCY_STOP,
                Permission.FIELD_GENERATE,
                Permission.FIELD_VIEW,
                Permission.SAFETY_VIEW,
                Permission.DATA_VIEW,
            },
            UserRole.OBSERVER: {
                Permission.FIELD_VIEW,
                Permission.SAFETY_VIEW,
                Permission.DATA_VIEW,
            },
            UserRole.GUEST: {
                Permission.FIELD_VIEW,
            },
        }
    
    def _setup_audit_logging(self, audit_log_path: str) -> logging.Logger:
        """Setup audit logging."""
        logger = logging.getLogger("audit")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(audit_log_path)
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _create_default_admin(self) -> None:
        """Create default admin user."""
        # Use environment variable or generate secure password
        admin_password = os.environ.get("ACOUSTO_ADMIN_PASSWORD")
        if not admin_password:
            # Generate secure random password for development
            import secrets
            import string
            alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
            admin_password = ''.join(secrets.choice(alphabet) for _ in range(16))
            print(f"⚠️  Generated admin password: {admin_password}")
            print("   Set ACOUSTO_ADMIN_PASSWORD environment variable for production")
        
        self.create_user(
            username="admin",
            password=admin_password,
            email="admin@example.com",
            role=UserRole.ADMIN
        )
        print(f"Default admin user created with password: {admin_password}")
        print("IMPORTANT: Change the admin password immediately!")
    
    def create_user(
        self,
        username: str,
        password: str,
        email: str,
        role: UserRole,
        permissions: Optional[Set[Permission]] = None
    ) -> bool:
        """
        Create new user account.
        
        Args:
            username: Unique username
            password: User password
            email: User email
            role: User role
            permissions: Custom permissions (uses role defaults if None)
            
        Returns:
            True if user created successfully
        """
        if username in self.users:
            return False
        
        if self.require_strong_passwords and not self._is_strong_password(password):
            raise ValueError("Password does not meet strength requirements")
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        
        # Set permissions
        if permissions is None:
            permissions = self.role_permissions.get(role, set())
        
        # Create user
        user = User(
            username=username,
            email=email,
            role=role,
            permissions=permissions,
            created_at=time.time()
        )
        
        self.users[username] = user
        self.password_hashes[username] = password_hash
        
        # Generate API key
        api_key = self._generate_api_key(username)
        user.api_key = api_key
        self.api_keys[api_key] = username
        
        self._log_audit(username, "USER_CREATED", f"role:{role.value}", "success")
        return True
    
    def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None
    ) -> Optional[str]:
        """
        Authenticate user and return session token.
        
        Args:
            username: Username
            password: Password
            ip_address: Client IP address for audit logging
            
        Returns:
            Session token if authentication successful, None otherwise
        """
        # Rate limiting
        if ip_address and not self._check_rate_limit(ip_address):
            self._log_audit(username, "LOGIN_ATTEMPT", "rate_limited", "failure", ip_address)
            return None
        
        if username not in self.users:
            self._log_audit(username, "LOGIN_ATTEMPT", "user_not_found", "failure", ip_address)
            return None
        
        user = self.users[username]
        
        # Check if account is locked
        if user.is_locked():
            self._log_audit(username, "LOGIN_ATTEMPT", "account_locked", "failure", ip_address)
            return None
        
        # Verify password
        if not bcrypt.checkpw(password.encode(), self.password_hashes[username]):
            user.failed_attempts += 1
            
            # Lock account if too many failed attempts
            if user.failed_attempts >= self.max_failed_attempts:
                user.locked_until = time.time() + self.lockout_duration
                self._log_audit(username, "ACCOUNT_LOCKED", f"failed_attempts:{user.failed_attempts}", "failure", ip_address)
            
            self._log_audit(username, "LOGIN_ATTEMPT", "invalid_password", "failure", ip_address)
            return None
        
        # Reset failed attempts on successful login
        user.failed_attempts = 0
        user.last_login = time.time()
        
        # Generate session token
        session_token = self._generate_session_token(username)
        user.session_token = session_token
        
        with self.session_lock:
            self.active_sessions[session_token] = username
        
        self._log_audit(username, "LOGIN_SUCCESS", "", "success", ip_address)
        return session_token
    
    def authenticate_api_key(self, api_key: str) -> Optional[str]:
        """
        Authenticate using API key.
        
        Args:
            api_key: API key
            
        Returns:
            Username if valid, None otherwise
        """
        if api_key in self.api_keys:
            username = self.api_keys[api_key]
            self._log_audit(username, "API_AUTH", "api_key", "success")
            return username
        
        self._log_audit("unknown", "API_AUTH", "invalid_key", "failure")
        return None
    
    def validate_session(self, token: str) -> Optional[str]:
        """
        Validate session token.
        
        Args:
            token: Session token
            
        Returns:
            Username if valid, None otherwise
        """
        with self.session_lock:
            if token not in self.active_sessions:
                return None
            
            username = self.active_sessions[token]
            user = self.users[username]
            
            # Check if token matches user's current token
            if user.session_token != token:
                return None
            
            # Check token expiry
            try:
                payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
                if payload['exp'] < time.time():
                    self._invalidate_session(token)
                    return None
                
                return username
                
            except jwt.InvalidTokenError:
                self._invalidate_session(token)
                return None
    
    def check_permission(
        self,
        username: str,
        permission: Permission,
        resource: str = ""
    ) -> bool:
        """
        Check if user has specific permission.
        
        Args:
            username: Username
            permission: Required permission
            resource: Resource being accessed (for audit)
            
        Returns:
            True if permission granted
        """
        if username not in self.users:
            self._log_audit(username, "PERMISSION_CHECK", f"{permission.value}:{resource}", "failure")
            return False
        
        user = self.users[username]
        has_perm = user.has_permission(permission)
        
        result = "success" if has_perm else "denied"
        self._log_audit(username, "PERMISSION_CHECK", f"{permission.value}:{resource}", result)
        
        return has_perm
    
    def require_permission(self, permission: Permission) -> Callable:
        """
        Decorator to require specific permission for function access.
        
        Args:
            permission: Required permission
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # In a real implementation, you'd extract the current user from context
                # For now, we'll assume it's passed as 'user' parameter
                current_user = kwargs.get('user')
                if not current_user or not self.check_permission(current_user, permission, func.__name__):
                    raise PermissionError(f"Permission {permission.value} required")
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def logout_user(self, token: str) -> bool:
        """
        Logout user by invalidating session.
        
        Args:
            token: Session token
            
        Returns:
            True if logout successful
        """
        username = self.validate_session(token)
        if username:
            self._invalidate_session(token)
            self._log_audit(username, "LOGOUT", "", "success")
            return True
        
        return False
    
    def change_password(
        self,
        username: str,
        old_password: str,
        new_password: str
    ) -> bool:
        """
        Change user password.
        
        Args:
            username: Username
            old_password: Current password
            new_password: New password
            
        Returns:
            True if password changed successfully
        """
        if username not in self.users:
            return False
        
        # Verify old password
        if not bcrypt.checkpw(old_password.encode(), self.password_hashes[username]):
            self._log_audit(username, "PASSWORD_CHANGE", "invalid_old_password", "failure")
            return False
        
        # Check new password strength
        if self.require_strong_passwords and not self._is_strong_password(new_password):
            self._log_audit(username, "PASSWORD_CHANGE", "weak_password", "failure")
            return False
        
        # Hash and store new password
        new_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt())
        self.password_hashes[username] = new_hash
        
        # Invalidate existing sessions for security
        user = self.users[username]
        if user.session_token:
            self._invalidate_session(user.session_token)
        
        self._log_audit(username, "PASSWORD_CHANGE", "", "success")
        return True
    
    def revoke_api_key(self, username: str) -> str:
        """
        Revoke and regenerate API key for user.
        
        Args:
            username: Username
            
        Returns:
            New API key
        """
        if username not in self.users:
            raise ValueError("User not found")
        
        user = self.users[username]
        
        # Remove old key
        if user.api_key and user.api_key in self.api_keys:
            del self.api_keys[user.api_key]
        
        # Generate new key
        new_key = self._generate_api_key(username)
        user.api_key = new_key
        self.api_keys[new_key] = username
        
        self._log_audit(username, "API_KEY_REVOKED", "", "success")
        return new_key
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information (excluding sensitive data)."""
        if username not in self.users:
            return None
        
        user = self.users[username]
        return {
            'username': user.username,
            'email': user.email,
            'role': user.role.value,
            'permissions': [p.value for p in user.permissions],
            'created_at': user.created_at,
            'last_login': user.last_login,
            'is_locked': user.is_locked(),
            'metadata': user.metadata
        }
    
    def list_users(self, requesting_user: str) -> List[Dict[str, Any]]:
        """List all users (requires USER_MANAGE permission)."""
        if not self.check_permission(requesting_user, Permission.USER_MANAGE):
            raise PermissionError("USER_MANAGE permission required")
        
        return [
            self.get_user_info(username) 
            for username in self.users.keys()
        ]
    
    def get_audit_log(
        self,
        requesting_user: str,
        hours: int = 24,
        user_filter: Optional[str] = None,
        action_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get audit log entries.
        
        Args:
            requesting_user: User making the request
            hours: Hours of history to retrieve
            user_filter: Filter by specific user
            action_filter: Filter by specific action
            
        Returns:
            List of audit events
        """
        if not self.check_permission(requesting_user, Permission.AUDIT_VIEW):
            raise PermissionError("AUDIT_VIEW permission required")
        
        cutoff_time = time.time() - hours * 3600
        
        filtered_events = []
        for event in self.audit_events:
            if event.timestamp < cutoff_time:
                continue
            
            if user_filter and event.user != user_filter:
                continue
            
            if action_filter and action_filter not in event.action:
                continue
            
            filtered_events.append({
                'timestamp': event.timestamp,
                'user': event.user,
                'action': event.action,
                'resource': event.resource,
                'result': event.result,
                'ip_address': event.ip_address,
                'metadata': event.metadata
            })
        
        return filtered_events
    
    def _generate_session_token(self, username: str) -> str:
        """Generate JWT session token."""
        payload = {
            'username': username,
            'iat': time.time(),
            'exp': time.time() + self.session_timeout
        }
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def _generate_api_key(self, username: str) -> str:
        """Generate API key for user."""
        # Create API key with username prefix for identification
        key_data = f"{username}:{secrets.token_urlsafe(32)}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _invalidate_session(self, token: str) -> None:
        """Invalidate session token."""
        with self.session_lock:
            if token in self.active_sessions:
                username = self.active_sessions[token]
                del self.active_sessions[token]
                
                # Clear user's session token
                if username in self.users:
                    self.users[username].session_token = None
    
    def _is_strong_password(self, password: str) -> bool:
        """Check if password meets strength requirements."""
        if len(password) < 8:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return has_upper and has_lower and has_digit and has_special
    
    def _check_rate_limit(self, ip_address: str) -> bool:
        """Check if IP address is within rate limit."""
        current_time = time.time()
        
        if ip_address not in self.rate_limits:
            self.rate_limits[ip_address] = []
        
        request_times = self.rate_limits[ip_address]
        
        # Remove old requests outside window
        request_times[:] = [t for t in request_times if current_time - t < self.rate_limit_window]
        
        # Check if within limit
        if len(request_times) >= self.rate_limit_max_requests:
            return False
        
        # Add current request
        request_times.append(current_time)
        return True
    
    def _log_audit(
        self,
        user: str,
        action: str,
        resource: str,
        result: str,
        ip_address: Optional[str] = None,
        **metadata
    ) -> None:
        """Log audit event."""
        event = AuditEvent(
            timestamp=time.time(),
            user=user,
            action=action,
            resource=resource,
            result=result,
            ip_address=ip_address,
            metadata=metadata
        )
        
        self.audit_events.append(event)
        
        # Also log to file
        log_message = f"{user} - {action} - {resource} - {result}"
        if ip_address:
            log_message += f" - {ip_address}"
        
        self.audit_logger.info(log_message)


# Security decorators for API endpoints
def require_auth(access_control: AccessControl):
    """Decorator to require authentication."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Extract token from request context (implementation dependent)
            token = kwargs.get('auth_token') or kwargs.get('session_token')
            if not token:
                raise PermissionError("Authentication required")
            
            username = access_control.validate_session(token)
            if not username:
                raise PermissionError("Invalid or expired token")
            
            kwargs['current_user'] = username
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(access_control: AccessControl, role: UserRole):
    """Decorator to require specific role."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if not current_user:
                raise PermissionError("Authentication required")
            
            user = access_control.users.get(current_user)
            if not user or user.role != role:
                raise PermissionError(f"Role {role.value} required")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator