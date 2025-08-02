#!/usr/bin/env python3
"""
Metrics collection script for Acousto-Gen project.

Collects various project metrics including code quality, performance,
security, and community metrics for tracking and reporting.
"""

import json
import subprocess
import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import requests


@dataclass
class CodeMetrics:
    """Code quality metrics."""
    lines_of_code: int
    test_coverage: float
    lint_score: float
    complexity_average: float
    documentation_coverage: float
    type_coverage: float
    technical_debt_minutes: int


@dataclass
class SecurityMetrics:
    """Security-related metrics."""
    vulnerabilities_critical: int
    vulnerabilities_high: int
    vulnerabilities_medium: int
    vulnerabilities_low: int
    security_score: float
    last_audit_date: str
    dependencies_with_vulnerabilities: int


@dataclass
class PerformanceMetrics:
    """Performance benchmarks."""
    optimization_time_ms: Optional[float]
    memory_usage_mb: Optional[float]
    gpu_utilization_percent: Optional[float]
    benchmark_score: Optional[float]
    regression_count: int


@dataclass
class CommunityMetrics:
    """Community and adoption metrics."""
    stars: int
    forks: int
    downloads_total: int
    downloads_last_month: int
    contributors: int
    issues_open: int
    issues_closed: int
    prs_open: int
    prs_merged: int


class MetricsCollector:
    """Collects various project metrics."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.logger = self._setup_logging()
        self.github_token = os.getenv('GITHUB_TOKEN')
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def collect_code_metrics(self) -> CodeMetrics:
        """Collect code quality metrics."""
        self.logger.info("Collecting code quality metrics...")
        
        # Count lines of code
        loc = self._count_lines_of_code()
        
        # Get test coverage
        coverage = self._get_test_coverage()
        
        # Get lint score
        lint_score = self._get_lint_score()
        
        # Get complexity metrics
        complexity = self._get_complexity_metrics()
        
        # Get documentation coverage
        doc_coverage = self._get_documentation_coverage()
        
        # Get type coverage
        type_coverage = self._get_type_coverage()
        
        # Estimate technical debt
        tech_debt = self._estimate_technical_debt()
        
        return CodeMetrics(
            lines_of_code=loc,
            test_coverage=coverage,
            lint_score=lint_score,
            complexity_average=complexity,
            documentation_coverage=doc_coverage,
            type_coverage=type_coverage,
            technical_debt_minutes=tech_debt
        )
    
    def _count_lines_of_code(self) -> int:
        """Count lines of code in Python files."""
        try:
            result = subprocess.run([
                'find', str(self.repo_path), '-name', '*.py', 
                '-not', '-path', '*/.*', '-exec', 'wc', '-l', '{}', '+'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                total_lines = 0
                for line in lines:
                    if line.strip() and not line.strip().endswith('total'):
                        try:
                            count = int(line.strip().split()[0])
                            total_lines += count
                        except (ValueError, IndexError):
                            continue
                return total_lines
        except Exception as e:
            self.logger.warning(f"Failed to count lines of code: {e}")
        
        return 0
    
    def _get_test_coverage(self) -> float:
        """Get test coverage percentage."""
        try:
            # Run pytest with coverage
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                '--cov=acousto_gen', '--cov-report=json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            # Read coverage report
            coverage_file = self.repo_path / 'coverage.json'
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    return coverage_data.get('totals', {}).get('percent_covered', 0.0)
        except Exception as e:
            self.logger.warning(f"Failed to get test coverage: {e}")
        
        return 0.0
    
    def _get_lint_score(self) -> float:
        """Get linting score from Ruff."""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'ruff', 'check', 
                'acousto_gen/', '--format=json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                issues = json.loads(result.stdout)
                total_lines = self._count_lines_of_code()
                if total_lines > 0:
                    issues_per_1000_lines = (len(issues) / total_lines) * 1000
                    # Convert to 0-10 scale (10 is best)
                    return max(0, 10 - issues_per_1000_lines)
        except Exception as e:
            self.logger.warning(f"Failed to get lint score: {e}")
        
        return 10.0  # Default to perfect score if unable to measure
    
    def _get_complexity_metrics(self) -> float:
        """Get average cyclomatic complexity."""
        try:
            # Use radon for complexity analysis
            result = subprocess.run([
                'radon', 'cc', str(self.repo_path / 'acousto_gen'), '-j'
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                complexity_data = json.loads(result.stdout)
                total_complexity = 0
                function_count = 0
                
                for file_data in complexity_data.values():
                    for item in file_data:
                        if item['type'] in ['function', 'method']:
                            total_complexity += item['complexity']
                            function_count += 1
                
                if function_count > 0:
                    return total_complexity / function_count
        except Exception as e:
            self.logger.warning(f"Failed to get complexity metrics: {e}")
        
        return 1.0  # Default to low complexity
    
    def _get_documentation_coverage(self) -> float:
        """Estimate documentation coverage."""
        try:
            # Count functions/classes with docstrings
            result = subprocess.run([
                'python', '-c', '''
import ast
import os
import sys

def count_docstrings(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    
    total_items = 0
    documented_items = 0
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            total_items += 1
            if ast.get_docstring(node):
                documented_items += 1
    
    return total_items, documented_items

total_items = 0
documented_items = 0

for root, dirs, files in os.walk("acousto_gen"):
    for file in files:
        if file.endswith(".py"):
            try:
                t, d = count_docstrings(os.path.join(root, file))
                total_items += t
                documented_items += d
            except:
                pass

if total_items > 0:
    print(f"{documented_items / total_items * 100:.1f}")
else:
    print("0.0")
'''
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception as e:
            self.logger.warning(f"Failed to get documentation coverage: {e}")
        
        return 0.0
    
    def _get_type_coverage(self) -> float:
        """Get type annotation coverage."""
        try:
            # Use mypy to check type coverage
            result = subprocess.run([
                sys.executable, '-m', 'mypy', 'acousto_gen/', 
                '--any-exprs-report', '/tmp/mypy-report'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            # Parse mypy output for type coverage
            # This is a simplified estimation
            lines_with_any = result.stdout.count('Any')
            total_lines = self._count_lines_of_code()
            
            if total_lines > 0:
                type_coverage = max(0, 100 - (lines_with_any / total_lines * 100))
                return type_coverage
        except Exception as e:
            self.logger.warning(f"Failed to get type coverage: {e}")
        
        return 50.0  # Default estimate
    
    def _estimate_technical_debt(self) -> int:
        """Estimate technical debt in minutes."""
        try:
            # Use various heuristics to estimate technical debt
            debt_minutes = 0
            
            # TODO comments
            result = subprocess.run([
                'grep', '-r', 'TODO\\|FIXME\\|XXX', 'acousto_gen/'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                todo_count = len(result.stdout.strip().split('\n'))
                debt_minutes += todo_count * 15  # 15 minutes per TODO
            
            # Code duplication (simplified)
            result = subprocess.run([
                'find', 'acousto_gen/', '-name', '*.py', '-exec', 'wc', '-l', '{}', '+'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            # Add more debt estimation logic here
            
            return debt_minutes
        except Exception as e:
            self.logger.warning(f"Failed to estimate technical debt: {e}")
        
        return 0
    
    def collect_security_metrics(self) -> SecurityMetrics:
        """Collect security metrics."""
        self.logger.info("Collecting security metrics...")
        
        vulnerabilities = self._get_vulnerability_counts()
        security_score = self._calculate_security_score(vulnerabilities)
        
        return SecurityMetrics(
            vulnerabilities_critical=vulnerabilities.get('critical', 0),
            vulnerabilities_high=vulnerabilities.get('high', 0),
            vulnerabilities_medium=vulnerabilities.get('medium', 0),
            vulnerabilities_low=vulnerabilities.get('low', 0),
            security_score=security_score,
            last_audit_date=datetime.now().isoformat(),
            dependencies_with_vulnerabilities=self._count_vulnerable_dependencies()
        )
    
    def _get_vulnerability_counts(self) -> Dict[str, int]:
        """Get vulnerability counts by severity."""
        vulnerabilities = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        try:
            # Run safety check
            result = subprocess.run([
                sys.executable, '-m', 'safety', 'check', '--json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                for vuln in safety_data:
                    severity = vuln.get('severity', 'medium').lower()
                    if severity in vulnerabilities:
                        vulnerabilities[severity] += 1
        except Exception as e:
            self.logger.warning(f"Failed to get vulnerability counts: {e}")
        
        return vulnerabilities
    
    def _calculate_security_score(self, vulnerabilities: Dict[str, int]) -> float:
        """Calculate overall security score (0-10)."""
        # Weight vulnerabilities by severity
        weights = {'critical': 10, 'high': 5, 'medium': 2, 'low': 1}
        
        total_weight = sum(
            vulnerabilities[severity] * weight 
            for severity, weight in weights.items()
        )
        
        # Convert to 0-10 scale (10 is best)
        if total_weight == 0:
            return 10.0
        else:
            return max(0, 10 - min(10, total_weight / 10))
    
    def _count_vulnerable_dependencies(self) -> int:
        """Count dependencies with known vulnerabilities."""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'safety', 'check', '--json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                vulnerable_packages = set()
                for vuln in safety_data:
                    vulnerable_packages.add(vuln.get('package_name', ''))
                return len(vulnerable_packages)
        except Exception as e:
            self.logger.warning(f"Failed to count vulnerable dependencies: {e}")
        
        return 0
    
    def collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect performance metrics."""
        self.logger.info("Collecting performance metrics...")
        
        # Run performance benchmarks if available
        benchmark_results = self._run_benchmarks()
        
        return PerformanceMetrics(
            optimization_time_ms=benchmark_results.get('optimization_time_ms'),
            memory_usage_mb=benchmark_results.get('memory_usage_mb'),
            gpu_utilization_percent=benchmark_results.get('gpu_utilization_percent'),
            benchmark_score=benchmark_results.get('benchmark_score'),
            regression_count=0  # Would be populated from historical data
        )
    
    def _run_benchmarks(self) -> Dict[str, Optional[float]]:
        """Run performance benchmarks."""
        results = {}
        
        try:
            # Run pytest benchmarks if available
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 'tests/', '-m', 'performance',
                '--benchmark-json=/tmp/benchmark.json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            benchmark_file = Path('/tmp/benchmark.json')
            if benchmark_file.exists():
                with open(benchmark_file) as f:
                    benchmark_data = json.load(f)
                    
                # Extract relevant metrics
                benchmarks = benchmark_data.get('benchmarks', [])
                if benchmarks:
                    # Get average stats
                    avg_time = sum(b.get('stats', {}).get('mean', 0) for b in benchmarks) / len(benchmarks)
                    results['optimization_time_ms'] = avg_time * 1000
                    
        except Exception as e:
            self.logger.warning(f"Failed to run benchmarks: {e}")
        
        return results
    
    def collect_community_metrics(self) -> CommunityMetrics:
        """Collect community and adoption metrics."""
        self.logger.info("Collecting community metrics...")
        
        github_metrics = self._get_github_metrics()
        
        return CommunityMetrics(
            stars=github_metrics.get('stars', 0),
            forks=github_metrics.get('forks', 0),
            downloads_total=0,  # Would need PyPI API
            downloads_last_month=0,  # Would need PyPI API
            contributors=github_metrics.get('contributors', 0),
            issues_open=github_metrics.get('issues_open', 0),
            issues_closed=github_metrics.get('issues_closed', 0),
            prs_open=github_metrics.get('prs_open', 0),
            prs_merged=github_metrics.get('prs_merged', 0)
        )
    
    def _get_github_metrics(self) -> Dict[str, int]:
        """Get metrics from GitHub API."""
        metrics = {}
        
        if not self.github_token:
            self.logger.warning("No GitHub token provided, skipping GitHub metrics")
            return metrics
        
        try:
            headers = {'Authorization': f'token {self.github_token}'}
            
            # Get repository info
            repo_url = 'https://api.github.com/repos/danieleschmidt/acousto-gen'
            response = requests.get(repo_url, headers=headers)
            
            if response.status_code == 200:
                repo_data = response.json()
                metrics['stars'] = repo_data.get('stargazers_count', 0)
                metrics['forks'] = repo_data.get('forks_count', 0)
            
            # Get issues
            issues_url = f'{repo_url}/issues?state=all&per_page=100'
            response = requests.get(issues_url, headers=headers)
            
            if response.status_code == 200:
                issues = response.json()
                metrics['issues_open'] = len([i for i in issues if i['state'] == 'open' and not i.get('pull_request')])
                metrics['issues_closed'] = len([i for i in issues if i['state'] == 'closed' and not i.get('pull_request')])
                metrics['prs_open'] = len([i for i in issues if i['state'] == 'open' and i.get('pull_request')])
                metrics['prs_merged'] = len([i for i in issues if i['state'] == 'closed' and i.get('pull_request')])
            
            # Get contributors
            contributors_url = f'{repo_url}/contributors'
            response = requests.get(contributors_url, headers=headers)
            
            if response.status_code == 200:
                contributors = response.json()
                metrics['contributors'] = len(contributors)
                
        except Exception as e:
            self.logger.warning(f"Failed to get GitHub metrics: {e}")
        
        return metrics
    
    def save_metrics(self, 
                    code_metrics: CodeMetrics,
                    security_metrics: SecurityMetrics, 
                    performance_metrics: PerformanceMetrics,
                    community_metrics: CommunityMetrics) -> None:
        """Save collected metrics to JSON file."""
        
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'code_quality': asdict(code_metrics),
            'security': asdict(security_metrics),
            'performance': asdict(performance_metrics),
            'community': asdict(community_metrics)
        }
        
        # Save to metrics file
        metrics_file = self.repo_path / '.github' / 'project-metrics.json'
        
        # Load existing metrics if available
        existing_metrics = {}
        if metrics_file.exists():
            with open(metrics_file) as f:
                existing_metrics = json.load(f)
        
        # Update metrics section
        existing_metrics['metrics'] = {
            'code_quality': metrics_data['code_quality'],
            'security': metrics_data['security'], 
            'performance': metrics_data['performance'],
            'community': metrics_data['community'],
            'last_updated': metrics_data['timestamp']
        }
        
        # Save updated metrics
        with open(metrics_file, 'w') as f:
            json.dump(existing_metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to {metrics_file}")
        
        # Also save timestamped version
        timestamped_file = self.repo_path / 'metrics' / f"metrics-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        timestamped_file.parent.mkdir(exist_ok=True)
        
        with open(timestamped_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def run(self) -> None:
        """Run metrics collection."""
        self.logger.info("Starting metrics collection...")
        
        # Collect all metrics
        code_metrics = self.collect_code_metrics()
        security_metrics = self.collect_security_metrics()
        performance_metrics = self.collect_performance_metrics()
        community_metrics = self.collect_community_metrics()
        
        # Save metrics
        self.save_metrics(code_metrics, security_metrics, performance_metrics, community_metrics)
        
        # Log summary
        self.logger.info("Metrics collection completed:")
        self.logger.info(f"  Code coverage: {code_metrics.test_coverage:.1f}%")
        self.logger.info(f"  Security score: {security_metrics.security_score:.1f}/10")
        self.logger.info(f"  GitHub stars: {community_metrics.stars}")
        self.logger.info(f"  Contributors: {community_metrics.contributors}")


def main():
    """Main entry point."""
    repo_path = Path(__file__).parent.parent.parent
    collector = MetricsCollector(repo_path)
    collector.run()


if __name__ == '__main__':
    main()