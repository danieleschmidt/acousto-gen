#!/usr/bin/env python3
"""
Security scan for Acousto-Gen codebase.
Performs static analysis for common security vulnerabilities.
"""

import os
import re
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple


class SecurityScanner:
    """Basic security scanner for Python code."""
    
    def __init__(self):
        """Initialize security scanner with vulnerability patterns."""
        self.vulnerability_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
            ],
            'sql_injection': [
                r'execute\s*\(\s*["\'].*%.*["\']',
                r'cursor\.execute\s*\(\s*["\'].*\+.*["\']',
                r'query\s*=\s*["\'].*\+.*["\']',
            ],
            'command_injection': [
                r'os\.system\s*\(',
                r'subprocess\.call\s*\(',
                r'subprocess\.run\s*\(',
                r'eval\s*\(',
                r'exec\s*\(',
            ],
            'path_traversal': [
                r'open\s*\(\s*.*\+.*\)',
                r'file\s*\(\s*.*\+.*\)',
                r'\.\./',
            ],
            'unsafe_deserialization': [
                r'pickle\.loads\s*\(',
                r'pickle\.load\s*\(',
                r'yaml\.load\s*\(',
                r'eval\s*\(',
            ],
            'weak_crypto': [
                r'hashlib\.md5',
                r'hashlib\.sha1',
                r'random\.random\(',
                r'random\.randint\(',
            ]
        }
        
        self.safe_patterns = {
            'safe_pickle': [
                r'# Security: Using pickle with trusted data only',
                r'# SECURITY REVIEWED',
                r'# Safe: controlled environment'
            ],
            'safe_subprocess': [
                r'shell=False',
                r'# Security: sanitized input',
            ]
        }
        
        self.findings: List[Dict[str, Any]] = []
    
    def scan_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan a single file for security vulnerabilities."""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check each vulnerability category
            for category, patterns in self.vulnerability_patterns.items():
                for pattern in patterns:
                    matches = list(re.finditer(pattern, content, re.IGNORECASE))
                    
                    for match in matches:
                        # Find line number
                        line_num = content[:match.start()].count('\n') + 1
                        line_content = lines[line_num - 1].strip()
                        
                        # Check if this is a false positive
                        is_false_positive = self._check_false_positive(
                            content, match, category, line_content
                        )
                        
                        if not is_false_positive:
                            findings.append({
                                'file': str(file_path),
                                'line': line_num,
                                'category': category,
                                'pattern': pattern,
                                'content': line_content,
                                'severity': self._get_severity(category),
                                'description': self._get_description(category)
                            })
        
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
        
        return findings
    
    def _check_false_positive(
        self,
        content: str,
        match: re.Match,
        category: str,
        line_content: str
    ) -> bool:
        """Check if a match is likely a false positive."""
        
        # Check for safety comments
        line_start = content.rfind('\n', 0, match.start()) + 1
        line_end = content.find('\n', match.end())
        if line_end == -1:
            line_end = len(content)
        
        context = content[max(0, line_start - 200):line_end + 200]
        
        # Check for safe patterns in context
        safe_patterns = self.safe_patterns.get(f'safe_{category.split("_")[0]}', [])
        for safe_pattern in safe_patterns:
            if re.search(safe_pattern, context, re.IGNORECASE):
                return True
        
        # Category-specific false positive checks
        if category == 'hardcoded_secrets':
            # Check for obvious test/example values
            test_indicators = ['test', 'example', 'demo', 'placeholder', 'dummy']
            if any(indicator in line_content.lower() for indicator in test_indicators):
                return True
            
            # Check for empty or obviously fake values
            match_text = match.group()
            if any(fake in match_text.lower() for fake in ['xxx', 'your_', 'replace_']):
                return True
        
        elif category == 'command_injection':
            # Check for safe subprocess usage
            if 'shell=False' in context or 'shell=False' in line_content:
                return True
        
        elif category == 'unsafe_deserialization':
            # Check for trusted data comments
            if 'trusted' in context.lower() or 'internal' in context.lower():
                return True
        
        return False
    
    def _get_severity(self, category: str) -> str:
        """Get severity level for vulnerability category."""
        severity_map = {
            'hardcoded_secrets': 'HIGH',
            'sql_injection': 'HIGH',
            'command_injection': 'HIGH',
            'path_traversal': 'MEDIUM',
            'unsafe_deserialization': 'HIGH',
            'weak_crypto': 'MEDIUM'
        }
        return severity_map.get(category, 'LOW')
    
    def _get_description(self, category: str) -> str:
        """Get description for vulnerability category."""
        descriptions = {
            'hardcoded_secrets': 'Hardcoded credentials or secrets found',
            'sql_injection': 'Potential SQL injection vulnerability',
            'command_injection': 'Potential command injection vulnerability',
            'path_traversal': 'Potential path traversal vulnerability',
            'unsafe_deserialization': 'Unsafe deserialization detected',
            'weak_crypto': 'Weak cryptographic function usage'
        }
        return descriptions.get(category, 'Security issue detected')
    
    def scan_directory(self, directory: Path) -> List[Dict[str, Any]]:
        """Scan all Python files in directory recursively."""
        all_findings = []
        
        for py_file in directory.rglob('*.py'):
            # Skip __pycache__ and test files for certain checks
            if '__pycache__' in str(py_file):
                continue
            
            findings = self.scan_file(py_file)
            all_findings.extend(findings)
        
        return all_findings
    
    def generate_report(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate security report from findings."""
        
        # Categorize findings by severity
        high_severity = [f for f in findings if f['severity'] == 'HIGH']
        medium_severity = [f for f in findings if f['severity'] == 'MEDIUM']
        low_severity = [f for f in findings if f['severity'] == 'LOW']
        
        # Count by category
        category_counts = {}
        for finding in findings:
            category = finding['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Generate summary
        summary = {
            'total_files_scanned': len(set(f['file'] for f in findings)),
            'total_issues_found': len(findings),
            'high_severity_issues': len(high_severity),
            'medium_severity_issues': len(medium_severity),
            'low_severity_issues': len(low_severity),
            'issues_by_category': category_counts,
            'scan_timestamp': str(Path.cwd()),
        }
        
        return {
            'summary': summary,
            'high_severity_findings': high_severity,
            'medium_severity_findings': medium_severity,
            'low_severity_findings': low_severity,
            'all_findings': findings
        }
    
    def print_report(self, report: Dict[str, Any]) -> None:
        """Print security scan report."""
        summary = report['summary']
        
        print("ACOUSTO-GEN SECURITY SCAN REPORT")
        print("=" * 50)
        print(f"Files scanned: {summary['total_files_scanned']}")
        print(f"Total issues found: {summary['total_issues_found']}")
        print(f"High severity: {summary['high_severity_issues']}")
        print(f"Medium severity: {summary['medium_severity_issues']}")
        print(f"Low severity: {summary['low_severity_issues']}")
        print()
        
        if summary['issues_by_category']:
            print("Issues by category:")
            for category, count in summary['issues_by_category'].items():
                print(f"  {category}: {count}")
            print()
        
        # Print high severity issues
        if report['high_severity_findings']:
            print("HIGH SEVERITY ISSUES:")
            print("-" * 30)
            for finding in report['high_severity_findings']:
                print(f"File: {finding['file']}")
                print(f"Line: {finding['line']}")
                print(f"Issue: {finding['description']}")
                print(f"Code: {finding['content']}")
                print()
        
        # Print medium severity issues (first 5)
        if report['medium_severity_findings']:
            print("MEDIUM SEVERITY ISSUES (first 5):")
            print("-" * 35)
            for finding in report['medium_severity_findings'][:5]:
                print(f"File: {finding['file']}")
                print(f"Line: {finding['line']}")
                print(f"Issue: {finding['description']}")
                print(f"Code: {finding['content']}")
                print()
            
            if len(report['medium_severity_findings']) > 5:
                print(f"... and {len(report['medium_severity_findings']) - 5} more")
                print()
        
        # Overall assessment
        if summary['high_severity_issues'] == 0:
            if summary['medium_severity_issues'] == 0:
                print("✓ SECURITY ASSESSMENT: PASS - No critical security issues found")
            else:
                print("⚠ SECURITY ASSESSMENT: REVIEW REQUIRED - Medium severity issues found")
        else:
            print("✗ SECURITY ASSESSMENT: FAIL - High severity issues must be addressed")


def main():
    """Run security scan on the codebase."""
    scanner = SecurityScanner()
    
    # Scan source directories
    scan_paths = [
        Path("src"),
        Path("acousto_gen"),
    ]
    
    all_findings = []
    files_scanned = 0
    
    for scan_path in scan_paths:
        if scan_path.exists():
            print(f"Scanning {scan_path}...")
            findings = scanner.scan_directory(scan_path)
            all_findings.extend(findings)
            
            # Count files
            files_scanned += len(list(scan_path.rglob('*.py')))
    
    # Generate and print report
    report = scanner.generate_report(all_findings)
    report['summary']['total_files_scanned'] = files_scanned
    
    scanner.print_report(report)
    
    # Save detailed report
    with open('security_scan_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: security_scan_report.json")
    
    # Return exit code based on findings
    high_severity_count = len(report['high_severity_findings'])
    return 1 if high_severity_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())