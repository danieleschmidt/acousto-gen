#!/usr/bin/env python3
"""
Acousto-Gen Security Scanner
Comprehensive security validation for production deployment.
"""

import sys
import os
import re
import hashlib
import json
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import warnings


@dataclass
class SecurityIssue:
    """Security issue data structure."""
    category: str
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    recommendation: str = ""
    cve_references: List[str] = None


class SecurityScanner:
    """Comprehensive security scanner for code and configuration."""
    
    def __init__(self, project_path: str = "."):
        """Initialize security scanner."""
        self.project_path = Path(project_path)
        self.issues: List[SecurityIssue] = []
        self.scanned_files = 0
        self.total_lines = 0
        
        # Security patterns to detect
        self.security_patterns = {
            'hardcoded_secrets': [
                (r'password\s*=\s*["\'][^"\']{8,}["\']', 'Potential hardcoded password'),
                (r'api[_-]?key\s*=\s*["\'][^"\']{16,}["\']', 'Potential hardcoded API key'),
                (r'secret[_-]?key\s*=\s*["\'][^"\']{16,}["\']', 'Potential hardcoded secret key'),
                (r'token\s*=\s*["\'][^"\']{20,}["\']', 'Potential hardcoded token'),
                (r'["\']sk_[a-zA-Z0-9]{32,}["\']', 'Potential Stripe secret key'),
                (r'["\']pk_[a-zA-Z0-9]{32,}["\']', 'Potential Stripe public key (info only)'),
            ],
            'sql_injection': [
                (r'\.execute\([^)]*%.*\)', 'Potential SQL injection vulnerability'),
                (r'\.format\([^)]*\).*execute', 'Potential SQL injection via format()'),
                (r'f["\'][^"\']*{[^}]*}[^"\']*["\'].*execute', 'Potential SQL injection via f-string'),
            ],
            'path_traversal': [
                (r'open\([^)]*\.\./', 'Potential path traversal vulnerability'),
                (r'os\.path\.join\([^)]*\.\./', 'Potential path traversal in path join'),
                (r'Path\([^)]*\.\./', 'Potential path traversal in Path constructor'),
            ],
            'command_injection': [
                (r'os\.system\([^)]*input', 'Potential command injection via os.system'),
                (r'subprocess\.[^(]*\([^)]*shell\s*=\s*True', 'Shell injection risk with shell=True'),
                (r'eval\([^)]*input', 'Potential code injection via eval()'),
                (r'exec\([^)]*input', 'Potential code injection via exec()'),
            ],
            'insecure_random': [
                (r'random\.(random|randint|choice)', 'Insecure random number generation'),
                (r'random\.Random\(\)', 'Insecure random number generator'),
            ],
            'unsafe_yaml': [
                (r'yaml\.load\([^)]*\)', 'Unsafe YAML loading - use safe_load()'),
                (r'yaml\.unsafe_load', 'Explicitly unsafe YAML loading'),
            ],
            'unsafe_pickle': [
                (r'pickle\.loads?\([^)]*\)', 'Unsafe pickle deserialization'),
                (r'cPickle\.loads?\([^)]*\)', 'Unsafe cPickle deserialization'),
            ],
            'debug_mode': [
                (r'debug\s*=\s*True', 'Debug mode enabled - disable in production'),
                (r'DEBUG\s*=\s*True', 'Debug mode enabled - disable in production'),
            ]
        }
        
        # File extensions to scan
        self.scannable_extensions = {'.py', '.js', '.ts', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'}
        
        # Critical files to check
        self.critical_files = ['requirements.txt', 'pyproject.toml', 'setup.py', 'Dockerfile', '.env']
    
    def scan_file(self, file_path: Path) -> None:
        """Scan a single file for security issues."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                self.total_lines += len(lines)
            
            self.scanned_files += 1
            
            # Check each security pattern category
            for category, patterns in self.security_patterns.items():
                for pattern, description in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        # Find line number
                        line_num = content[:match.start()].count('\n') + 1
                        
                        # Determine severity
                        severity = self._get_severity(category)
                        
                        issue = SecurityIssue(
                            category=category,
                            severity=severity,
                            description=description,
                            file_path=str(file_path.relative_to(self.project_path)),
                            line_number=line_num,
                            recommendation=self._get_recommendation(category),
                            cve_references=self._get_cve_references(category)
                        )
                        
                        self.issues.append(issue)
        
        except Exception as e:
            # Log scanning error but continue
            issue = SecurityIssue(
                category='scan_error',
                severity='info',
                description=f'Could not scan file: {e}',
                file_path=str(file_path.relative_to(self.project_path))
            )
            self.issues.append(issue)
    
    def _get_severity(self, category: str) -> str:
        """Get severity level for security category."""
        severity_map = {
            'hardcoded_secrets': 'critical',
            'sql_injection': 'critical', 
            'command_injection': 'critical',
            'path_traversal': 'high',
            'unsafe_yaml': 'high',
            'unsafe_pickle': 'high',
            'insecure_random': 'medium',
            'debug_mode': 'medium',
            'scan_error': 'info'
        }
        return severity_map.get(category, 'medium')
    
    def _get_recommendation(self, category: str) -> str:
        """Get security recommendation for category."""
        recommendations = {
            'hardcoded_secrets': 'Use environment variables or secure secret management systems',
            'sql_injection': 'Use parameterized queries or ORM with proper escaping',
            'command_injection': 'Validate input and avoid shell=True, use subprocess with list args',
            'path_traversal': 'Validate and sanitize file paths, use Path.resolve()',
            'unsafe_yaml': 'Use yaml.safe_load() instead of yaml.load()',
            'unsafe_pickle': 'Use safe serialization formats like JSON, avoid pickle for untrusted data',
            'insecure_random': 'Use secrets module or os.urandom() for cryptographic randomness',
            'debug_mode': 'Disable debug mode in production environments'
        }
        return recommendations.get(category, 'Review and mitigate security risk')
    
    def _get_cve_references(self, category: str) -> List[str]:
        """Get CVE references for security category."""
        cve_map = {
            'sql_injection': ['CWE-89', 'OWASP-A03'],
            'command_injection': ['CWE-78', 'OWASP-A03'],
            'path_traversal': ['CWE-22', 'OWASP-A01'],
            'unsafe_yaml': ['CVE-2017-18342', 'CWE-502'],
            'unsafe_pickle': ['CWE-502', 'OWASP-A08'],
            'hardcoded_secrets': ['CWE-798', 'OWASP-A02']
        }
        return cve_map.get(category, [])
    
    def scan_directory(self) -> None:
        """Scan entire project directory."""
        print(f"🔍 Scanning {self.project_path} for security issues...")
        
        for root, dirs, files in os.walk(self.project_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv'}]
            
            for file in files:
                file_path = Path(root) / file
                
                # Check if file should be scanned
                if (file_path.suffix in self.scannable_extensions or 
                    file in self.critical_files or
                    file_path.name.startswith('.')):
                    
                    self.scan_file(file_path)
    
    def check_dependencies(self) -> None:
        """Check for known vulnerable dependencies."""
        print("🔍 Checking dependencies for known vulnerabilities...")
        
        # Check requirements.txt
        req_file = self.project_path / 'requirements.txt'
        if req_file.exists():
            self._check_requirements_file(req_file)
        
        # Check pyproject.toml
        pyproject_file = self.project_path / 'pyproject.toml'
        if pyproject_file.exists():
            self._check_pyproject_file(pyproject_file)
    
    def _check_requirements_file(self, req_file: Path) -> None:
        """Check requirements.txt for vulnerable packages."""
        try:
            with open(req_file, 'r') as f:
                requirements = f.read()
            
            # Known vulnerable patterns (this would normally query a vulnerability database)
            vulnerable_packages = {
                'requests': ('2.25.0', 'CVE-2021-33503', 'Update to requests >= 2.25.1'),
                'pillow': ('8.1.0', 'CVE-2021-25287', 'Update to Pillow >= 8.1.2'),
                'urllib3': ('1.26.3', 'CVE-2021-33503', 'Update to urllib3 >= 1.26.4'),
            }
            
            for package, (version, cve, fix) in vulnerable_packages.items():
                if package in requirements.lower():
                    issue = SecurityIssue(
                        category='vulnerable_dependency',
                        severity='high',
                        description=f'Potentially vulnerable package: {package}',
                        file_path=str(req_file.relative_to(self.project_path)),
                        recommendation=fix,
                        cve_references=[cve]
                    )
                    self.issues.append(issue)
        
        except Exception as e:
            print(f"Warning: Could not check {req_file}: {e}")
    
    def _check_pyproject_file(self, pyproject_file: Path) -> None:
        """Check pyproject.toml for security issues."""
        try:
            with open(pyproject_file, 'r') as f:
                content = f.read()
            
            # Check for unsafe configurations
            if 'allow-direct-references = true' in content.lower():
                issue = SecurityIssue(
                    category='unsafe_config',
                    severity='medium',
                    description='Direct references allowed in pyproject.toml',
                    file_path=str(pyproject_file.relative_to(self.project_path)),
                    recommendation='Disable direct references for production'
                )
                self.issues.append(issue)
        
        except Exception as e:
            print(f"Warning: Could not check {pyproject_file}: {e}")
    
    def check_file_permissions(self) -> None:
        """Check for unsafe file permissions."""
        print("🔍 Checking file permissions...")
        
        for root, dirs, files in os.walk(self.project_path):
            for file in files:
                file_path = Path(root) / file
                
                try:
                    stat_info = file_path.stat()
                    mode = stat_info.st_mode
                    
                    # Check for world-writable files
                    if mode & 0o002:
                        issue = SecurityIssue(
                            category='unsafe_permissions',
                            severity='medium',
                            description='World-writable file detected',
                            file_path=str(file_path.relative_to(self.project_path)),
                            recommendation='Remove world-write permissions: chmod o-w filename'
                        )
                        self.issues.append(issue)
                    
                    # Check for overly permissive executable files
                    if (mode & 0o111) and (mode & 0o077):  # Executable and group/other has permissions
                        if file_path.suffix in {'.py', '.sh', '.bash'}:
                            issue = SecurityIssue(
                                category='unsafe_permissions',
                                severity='low',
                                description='Executable file with broad permissions',
                                file_path=str(file_path.relative_to(self.project_path)),
                                recommendation='Restrict permissions: chmod 755 or 744'
                            )
                            self.issues.append(issue)
                
                except (OSError, PermissionError):
                    # Skip files we can't read
                    continue
    
    def check_environment_files(self) -> None:
        """Check for sensitive files that shouldn't be committed."""
        print("🔍 Checking for sensitive environment files...")
        
        sensitive_files = ['.env', '.env.local', '.env.production', 'config.ini', 'secrets.json', '.secrets']
        sensitive_patterns = ['*secret*', '*password*', '*token*', '*key*', '.env*']
        
        for pattern in sensitive_patterns:
            for file_path in self.project_path.rglob(pattern):
                if file_path.is_file():
                    # Check if it's in version control
                    gitignore_path = self.project_path / '.gitignore'
                    
                    is_ignored = False
                    if gitignore_path.exists():
                        with open(gitignore_path, 'r') as f:
                            gitignore_content = f.read()
                            relative_path = str(file_path.relative_to(self.project_path))
                            
                            # Simple gitignore check
                            for line in gitignore_content.split('\n'):
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    if line in relative_path or relative_path.endswith(line):
                                        is_ignored = True
                                        break
                    
                    if not is_ignored:
                        issue = SecurityIssue(
                            category='sensitive_file',
                            severity='high',
                            description='Potentially sensitive file not in .gitignore',
                            file_path=str(file_path.relative_to(self.project_path)),
                            recommendation='Add to .gitignore or remove if contains secrets'
                        )
                        self.issues.append(issue)
    
    def run_comprehensive_scan(self) -> Dict[str, Any]:
        """Run complete security scan."""
        print("🛡️  Starting Comprehensive Security Scan")
        print("=" * 60)
        
        # Run all scan components
        self.scan_directory()
        self.check_dependencies()
        self.check_file_permissions()
        self.check_environment_files()
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        # Categorize issues by severity
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'info': 0}
        category_counts = {}
        
        for issue in self.issues:
            severity_counts[issue.severity] += 1
            category_counts[issue.category] = category_counts.get(issue.category, 0) + 1
        
        # Calculate security score (0-100)
        total_weighted_issues = (
            severity_counts['critical'] * 10 +
            severity_counts['high'] * 5 +
            severity_counts['medium'] * 2 +
            severity_counts['low'] * 1
        )
        
        # Base score starts at 100, deduct for issues
        security_score = max(0, 100 - total_weighted_issues)
        
        # Determine overall risk level
        if severity_counts['critical'] > 0:
            risk_level = 'CRITICAL'
        elif severity_counts['high'] > 0:
            risk_level = 'HIGH'
        elif severity_counts['medium'] > 0:
            risk_level = 'MEDIUM'
        elif severity_counts['low'] > 0:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'
        
        report = {
            'scan_summary': {
                'files_scanned': self.scanned_files,
                'lines_scanned': self.total_lines,
                'total_issues': len(self.issues),
                'security_score': security_score,
                'risk_level': risk_level
            },
            'severity_breakdown': severity_counts,
            'category_breakdown': category_counts,
            'issues': [
                {
                    'category': issue.category,
                    'severity': issue.severity,
                    'description': issue.description,
                    'file': issue.file_path,
                    'line': issue.line_number,
                    'recommendation': issue.recommendation,
                    'cve_references': issue.cve_references or []
                }
                for issue in self.issues
            ]
        }
        
        return report
    
    def print_report(self, report: Dict[str, Any]) -> None:
        """Print human-readable security report."""
        summary = report['scan_summary']
        
        print(f"\n📊 SECURITY SCAN RESULTS")
        print("=" * 60)
        
        # Overall status
        risk_icon = {
            'MINIMAL': '✅',
            'LOW': '🟨', 
            'MEDIUM': '🟠',
            'HIGH': '🔴',
            'CRITICAL': '🚨'
        }.get(summary['risk_level'], '❓')
        
        print(f"{risk_icon} RISK LEVEL: {summary['risk_level']}")
        print(f"🏆 Security Score: {summary['security_score']}/100")
        print(f"📁 Files Scanned: {summary['files_scanned']}")
        print(f"📄 Lines Analyzed: {summary['lines_scanned']:,}")
        print(f"🚨 Total Issues: {summary['total_issues']}")
        
        # Severity breakdown
        if summary['total_issues'] > 0:
            print(f"\n🔍 SEVERITY BREAKDOWN:")
            severity_counts = report['severity_breakdown']
            for severity in ['critical', 'high', 'medium', 'low', 'info']:
                count = severity_counts[severity]
                if count > 0:
                    icon = {'critical': '🚨', 'high': '🔴', 'medium': '🟠', 'low': '🟨', 'info': '🔵'}[severity]
                    print(f"   {icon} {severity.title()}: {count}")
            
            # Category breakdown
            print(f"\n📋 ISSUE CATEGORIES:")
            for category, count in report['category_breakdown'].items():
                print(f"   • {category.replace('_', ' ').title()}: {count}")
            
            # Critical and high issues
            critical_high_issues = [
                issue for issue in report['issues'] 
                if issue['severity'] in ['critical', 'high']
            ]
            
            if critical_high_issues:
                print(f"\n🚨 CRITICAL & HIGH SEVERITY ISSUES:")
                print("-" * 60)
                for i, issue in enumerate(critical_high_issues[:10], 1):  # Show top 10
                    severity_icon = '🚨' if issue['severity'] == 'critical' else '🔴'
                    location = f"{issue['file']}:{issue['line']}" if issue['line'] else issue['file']
                    
                    print(f"{i}. {severity_icon} {issue['description']}")
                    print(f"   📁 Location: {location}")
                    print(f"   💡 Fix: {issue['recommendation']}")
                    if issue['cve_references']:
                        print(f"   🔗 References: {', '.join(issue['cve_references'])}")
                    print()
                
                if len(critical_high_issues) > 10:
                    print(f"   ... and {len(critical_high_issues) - 10} more critical/high issues")
        
        else:
            print(f"\n✅ No security issues found!")
        
        # Recommendations
        print(f"\n💡 SECURITY RECOMMENDATIONS:")
        print("-" * 60)
        
        if summary['security_score'] < 70:
            print("• 🚨 URGENT: Address critical and high severity issues immediately")
        
        if summary['risk_level'] in ['HIGH', 'CRITICAL']:
            print("• 🔒 Review all hardcoded secrets and move to secure storage")
            print("• 🛡️  Implement input validation and sanitization")
        
        print("• 🔍 Regular security scans and dependency updates")
        print("• 📋 Security code review process")
        print("• 🔐 Principle of least privilege for file permissions")
        print("• 🌍 Security headers and HTTPS in production")
        
        print("\n" + "=" * 60)


def run_security_scan(project_path: str = ".") -> Dict[str, Any]:
    """Run comprehensive security scan on project."""
    scanner = SecurityScanner(project_path)
    report = scanner.run_comprehensive_scan()
    scanner.print_report(report)
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Acousto-Gen Security Scanner")
    parser.add_argument("--path", default=".", help="Project path to scan")
    parser.add_argument("--output", help="Output report to JSON file")
    
    args = parser.parse_args()
    
    # Run security scan
    report = run_security_scan(args.path)
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to: {args.output}")
    
    # Exit with appropriate code
    risk_level = report['scan_summary']['risk_level']
    if risk_level in ['CRITICAL', 'HIGH']:
        sys.exit(1)  # Fail for high-risk issues
    else:
        sys.exit(0)  # Success for low/minimal risk