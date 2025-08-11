#!/usr/bin/env python3
"""
Deployment verification script for Acousto-Gen.
Performs comprehensive checks to ensure production readiness.
"""

import os
import sys
import importlib
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
repo_root = Path(__file__).parent
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))

class DeploymentChecker:
    """Comprehensive deployment readiness checker."""
    
    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def check(self, name: str, condition: bool, message: str = "", warning: bool = False):
        """Register a check result."""
        status = "✅" if condition else ("⚠️" if warning else "❌")
        
        if condition:
            self.passed += 1
        elif warning:
            self.warnings += 1
        else:
            self.failed += 1
        
        result = f"{status} {name}"
        if message:
            result += f": {message}"
        
        print(result)
        self.checks.append((name, condition, message, warning))
        return condition
    
    def check_imports(self):
        """Check critical module imports."""
        print("📦 Checking Module Imports")
        print("-" * 30)
        
        # Core modules
        try:
            from acousto_gen import AcousticHologram
            self.check("Core AcousticHologram", True)
        except ImportError as e:
            self.check("Core AcousticHologram", False, str(e))
        
        try:
            from acousto_gen.cli import app
            self.check("CLI Module", True)
        except ImportError as e:
            self.check("CLI Module", False, str(e))
        
        # Validation
        try:
            from validation.input_validator import AcousticParameterValidator
            self.check("Input Validation", True)
        except ImportError as e:
            self.check("Input Validation", False, str(e))
        
        # Security
        try:
            from security.access_control import AccessControl
            self.check("Security Module", True)
        except ImportError as e:
            self.check("Security Module", False, str(e))
        
        # Monitoring
        try:
            from monitoring.metrics import MetricsCollector
            self.check("Metrics Collection", True)
        except ImportError as e:
            self.check("Metrics Collection", False, str(e))
        
        try:
            from monitoring.health_checks import HealthChecker
            self.check("Health Checks", True)
        except ImportError as e:
            self.check("Health Checks", False, str(e))
        
        # Optimization
        try:
            from optimization.cache import AcousticComputationCache
            self.check("Caching System", True)
        except ImportError as e:
            self.check("Caching System", False, str(e))
        
        try:
            from optimization.parallel_computing import ResourceManager
            self.check("Parallel Computing", True)
        except ImportError as e:
            self.check("Parallel Computing", False, str(e))
        
        # Infrastructure
        try:
            from infrastructure.auto_scaling import AutoScaler, LoadBalancer
            self.check("Auto-scaling & Load Balancing", True)
        except ImportError as e:
            self.check("Auto-scaling & Load Balancing", False, str(e))
        
        # Logging
        try:
            from logging.logger_config import initialize_logging
            self.check("Logging Configuration", True)
        except ImportError as e:
            self.check("Logging Configuration", False, str(e))
    
    def check_file_structure(self):
        """Check required file structure."""
        print("\\n📁 Checking File Structure")
        print("-" * 30)
        
        required_files = [
            "acousto_gen/__init__.py",
            "acousto_gen/core.py",
            "acousto_gen/cli.py",
            "src/validation/input_validator.py",
            "src/security/access_control.py",
            "src/monitoring/metrics.py",
            "src/monitoring/health_checks.py",
            "src/optimization/cache.py",
            "src/optimization/parallel_computing.py",
            "src/infrastructure/auto_scaling.py",
            "src/logging/logger_config.py"
        ]
        
        for file_path in required_files:
            full_path = repo_root / file_path
            exists = full_path.exists()
            self.check(f"File: {file_path}", exists)
    
    def check_configuration(self):
        """Check configuration files and settings."""
        print("\\n⚙️ Checking Configuration")
        print("-" * 30)
        
        # Check for pyproject.toml
        pyproject_path = repo_root / "pyproject.toml"
        self.check("pyproject.toml exists", pyproject_path.exists())
        
        # Check for requirements files
        req_paths = [
            repo_root / "requirements.txt",
            repo_root / "requirements-dev.txt"
        ]
        
        for req_path in req_paths:
            exists = req_path.exists()
            self.check(f"{req_path.name} exists", exists, warning=not exists)
        
        # Check for environment variables
        important_env_vars = [
            "ACOUSTO_ADMIN_PASSWORD"
        ]
        
        for env_var in important_env_vars:
            exists = env_var in os.environ
            self.check(f"Environment variable: {env_var}", exists, warning=not exists)
    
    def check_security(self):
        """Check security configurations."""
        print("\\n🔒 Checking Security")
        print("-" * 30)
        
        # Check for hardcoded secrets
        sensitive_patterns = ["password", "secret", "key", "token"]
        config_files = list(repo_root.glob("**/*.py"))
        
        has_hardcoded_secrets = False
        for file_path in config_files[:10]:  # Sample check
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    for pattern in sensitive_patterns:
                        if f"{pattern} = " in content and "example" not in content:
                            has_hardcoded_secrets = True
                            break
                if has_hardcoded_secrets:
                    break
            except:
                continue
        
        self.check("No hardcoded secrets", not has_hardcoded_secrets)
        
        # Check permissions (basic)
        secure_files = [
            repo_root / "src" / "security" / "access_control.py"
        ]
        
        for file_path in secure_files:
            if file_path.exists():
                stat = file_path.stat()
                is_secure = oct(stat.st_mode)[-3:] in ['644', '640', '600']
                self.check(f"Secure permissions: {file_path.name}", is_secure, warning=not is_secure)
    
    def check_functionality(self):
        """Check basic functionality."""
        print("\\n🔧 Checking Basic Functionality")
        print("-" * 30)
        
        # Test validation
        try:
            from validation.input_validator import AcousticParameterValidator
            validator = AcousticParameterValidator()
            
            # Test frequency validation
            freq = validator.validate_frequency(40000)
            self.check("Frequency validation", freq == 40000)
            
            # Test position validation
            pos = validator.validate_position([0, 0, 0.1])
            self.check("Position validation", len(pos) == 3)
            
        except Exception as e:
            self.check("Validation functionality", False, str(e))
        
        # Test security
        try:
            from security.access_control import AccessControl
            ac = AccessControl()
            
            # Test user creation
            created = ac.create_user("test_user", "TestPass123!", "test@example.com", ac.UserRole.GUEST)
            self.check("User creation", created)
            
        except Exception as e:
            self.check("Security functionality", False, str(e))
        
        # Test caching
        try:
            from optimization.cache import LRUCache
            cache = LRUCache(max_size=10, max_memory_mb=1)
            
            cache.put("test_key", "test_value")
            value = cache.get("test_key")
            self.check("Cache functionality", value == "test_value")
            
        except Exception as e:
            self.check("Cache functionality", False, str(e))
    
    def check_performance(self):
        """Check performance characteristics."""
        print("\\n🚀 Checking Performance")
        print("-" * 30)
        
        try:
            # Test import time
            start_time = time.time()
            from validation.input_validator import AcousticParameterValidator
            import_time = time.time() - start_time
            
            self.check("Fast imports", import_time < 1.0, f"{import_time:.3f}s")
            
            # Test validation speed
            validator = AcousticParameterValidator()
            start_time = time.time()
            
            for i in range(100):
                validator.validate_frequency(40000 + i)
            
            validation_time = time.time() - start_time
            rate = 100 / validation_time
            
            self.check("Validation performance", rate > 1000, f"{rate:.0f} validations/sec")
            
        except Exception as e:
            self.check("Performance tests", False, str(e))
    
    def run_all_checks(self):
        """Run all deployment checks."""
        print("🚀 Acousto-Gen Deployment Verification")
        print("=" * 50)
        
        self.check_imports()
        self.check_file_structure()
        self.check_configuration()
        self.check_security()
        self.check_functionality()
        self.check_performance()
        
        print("\\n" + "=" * 50)
        print("📊 DEPLOYMENT SUMMARY")
        print("=" * 50)
        print(f"✅ Passed:   {self.passed}")
        print(f"⚠️  Warnings: {self.warnings}")
        print(f"❌ Failed:   {self.failed}")
        print(f"📈 Total:    {len(self.checks)}")
        
        if self.failed == 0:
            print("\\n🎉 DEPLOYMENT READY!")
            print("System passed all critical checks.")
            return True
        else:
            print("\\n⚠️  DEPLOYMENT ISSUES DETECTED")
            print(f"Please address {self.failed} failed checks before deployment.")
            return False


if __name__ == "__main__":
    checker = DeploymentChecker()
    ready = checker.run_all_checks()
    
    sys.exit(0 if ready else 1)