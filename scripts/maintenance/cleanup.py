#!/usr/bin/env python3
"""
Repository maintenance and cleanup script for Acousto-Gen.

Performs various maintenance tasks including cleanup of old files,
optimization of repository structure, and health checks.
"""

import os
import shutil
import subprocess
import sys
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json


class RepositoryMaintenance:
    """Handles repository maintenance tasks."""
    
    def __init__(self, repo_path: Path, dry_run: bool = False):
        self.repo_path = repo_path
        self.dry_run = dry_run
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def cleanup_cache_files(self) -> None:
        """Clean up various cache files and directories."""
        self.logger.info("Cleaning up cache files...")
        
        cache_patterns = [
            '**/__pycache__',
            '**/*.pyc',
            '**/*.pyo',
            '**/.pytest_cache',
            '**/.mypy_cache',
            '**/.ruff_cache',
            '**/htmlcov',
            '**/.coverage',
            '**/.coverage.*',
            '**/build',
            '**/dist',
            '**/*.egg-info',
            '**/.tox',
            '**/.nox',
            '**/node_modules',
            '**/.DS_Store',
            '**/Thumbs.db'
        ]
        
        removed_count = 0
        for pattern in cache_patterns:
            for path in self.repo_path.glob(pattern):
                if path.exists():
                    if self.dry_run:
                        self.logger.info(f"[DRY RUN] Would remove: {path}")
                    else:
                        try:
                            if path.is_dir():
                                shutil.rmtree(path)
                            else:
                                path.unlink()
                            removed_count += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to remove {path}: {e}")
        
        self.logger.info(f"Cleaned up {removed_count} cache files/directories")
    
    def cleanup_log_files(self, max_age_days: int = 30) -> None:
        """Clean up old log files."""
        self.logger.info(f"Cleaning up log files older than {max_age_days} days...")
        
        log_patterns = [
            '**/logs/*.log',
            '**/hardware_logs/*.log',
            '**/safety_logs/*.log',
            '**/*.log.*',
            '**/benchmark-*.json',
            '**/profile-*.prof'
        ]
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        removed_count = 0
        
        for pattern in log_patterns:
            for log_file in self.repo_path.glob(pattern):
                try:
                    if log_file.is_file():
                        file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                        if file_time < cutoff_date:
                            if self.dry_run:
                                self.logger.info(f"[DRY RUN] Would remove old log: {log_file}")
                            else:
                                log_file.unlink()
                                removed_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to process log file {log_file}: {e}")
        
        self.logger.info(f"Cleaned up {removed_count} old log files")
    
    def cleanup_temporary_files(self) -> None:
        """Clean up temporary files."""
        self.logger.info("Cleaning up temporary files...")
        
        temp_patterns = [
            '**/*.tmp',
            '**/*.temp',
            '**/.tmp*',
            '**/tmp_*',
            '**/*~',
            '**/*.swp',
            '**/*.swo',
            '**/#*#',
            '**/.#*'
        ]
        
        removed_count = 0
        for pattern in temp_patterns:
            for temp_file in self.repo_path.glob(pattern):
                if temp_file.is_file():
                    if self.dry_run:
                        self.logger.info(f"[DRY RUN] Would remove temp file: {temp_file}")
                    else:
                        try:
                            temp_file.unlink()
                            removed_count += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to remove temp file {temp_file}: {e}")
        
        self.logger.info(f"Cleaned up {removed_count} temporary files")
    
    def optimize_git_repository(self) -> None:
        """Optimize git repository structure."""
        self.logger.info("Optimizing git repository...")
        
        if self.dry_run:
            self.logger.info("[DRY RUN] Would run git optimization commands")
            return
        
        try:
            # Run git garbage collection
            subprocess.run(['git', 'gc', '--aggressive'], 
                         cwd=self.repo_path, check=True)
            
            # Prune remote tracking branches
            subprocess.run(['git', 'remote', 'prune', 'origin'], 
                         cwd=self.repo_path, check=True)
            
            # Cleanup unnecessary files and optimize local repository
            subprocess.run(['git', 'fsck'], cwd=self.repo_path, check=True)
            
            self.logger.info("Git repository optimized successfully")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to optimize git repository: {e}")
    
    def check_large_files(self, size_limit_mb: int = 10) -> List[Path]:
        """Check for large files in the repository."""
        self.logger.info(f"Checking for files larger than {size_limit_mb}MB...")
        
        large_files = []
        size_limit_bytes = size_limit_mb * 1024 * 1024
        
        for file_path in self.repo_path.rglob('*'):
            if file_path.is_file():
                try:
                    # Skip files in .git directory
                    if '.git' in file_path.parts:
                        continue
                        
                    file_size = file_path.stat().st_size
                    if file_size > size_limit_bytes:
                        large_files.append(file_path)
                        self.logger.warning(
                            f"Large file found: {file_path} "
                            f"({file_size / (1024*1024):.1f}MB)"
                        )
                except Exception as e:
                    self.logger.debug(f"Could not check size of {file_path}: {e}")
        
        if large_files:
            self.logger.warning(f"Found {len(large_files)} large files")
        else:
            self.logger.info("No large files found")
            
        return large_files
    
    def validate_file_permissions(self) -> None:
        """Validate and fix file permissions."""
        self.logger.info("Validating file permissions...")
        
        # Python files should be readable but not executable
        python_files = list(self.repo_path.glob('**/*.py'))
        
        fixed_count = 0
        for py_file in python_files:
            try:
                current_mode = py_file.stat().st_mode & 0o777
                expected_mode = 0o644  # rw-r--r--
                
                if current_mode != expected_mode:
                    if self.dry_run:
                        self.logger.info(f"[DRY RUN] Would fix permissions for: {py_file}")
                    else:
                        py_file.chmod(expected_mode)
                        fixed_count += 1
                        
            except Exception as e:
                self.logger.warning(f"Failed to check permissions for {py_file}: {e}")
        
        # Scripts should be executable
        script_dirs = ['scripts']
        for script_dir in script_dirs:
            script_path = self.repo_path / script_dir
            if script_path.exists():
                for script_file in script_path.glob('**/*.py'):
                    try:
                        current_mode = script_file.stat().st_mode & 0o777
                        expected_mode = 0o755  # rwxr-xr-x
                        
                        if current_mode != expected_mode:
                            if self.dry_run:
                                self.logger.info(f"[DRY RUN] Would make executable: {script_file}")
                            else:
                                script_file.chmod(expected_mode)
                                fixed_count += 1
                                
                    except Exception as e:
                        self.logger.warning(f"Failed to fix script permissions for {script_file}: {e}")
        
        if fixed_count > 0:
            self.logger.info(f"Fixed permissions for {fixed_count} files")
        else:
            self.logger.info("All file permissions are correct")
    
    def check_dependency_health(self) -> Dict[str, Any]:
        """Check health of project dependencies."""
        self.logger.info("Checking dependency health...")
        
        health_report = {
            'outdated_packages': [],
            'vulnerable_packages': [],
            'total_packages': 0,
            'health_score': 10.0
        }
        
        try:
            # Check for outdated packages
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0 and result.stdout:
                outdated = json.loads(result.stdout)
                health_report['outdated_packages'] = outdated
                
        except Exception as e:
            self.logger.warning(f"Failed to check outdated packages: {e}")
        
        try:
            # Check for vulnerable packages
            result = subprocess.run([
                sys.executable, '-m', 'safety', 'check', '--json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                vulnerabilities = json.loads(result.stdout)
                health_report['vulnerable_packages'] = vulnerabilities
                
        except Exception as e:
            self.logger.warning(f"Failed to check package vulnerabilities: {e}")
        
        try:
            # Count total packages
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'list', '--format=json'
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0 and result.stdout:
                packages = json.loads(result.stdout)
                health_report['total_packages'] = len(packages)
                
        except Exception as e:
            self.logger.warning(f"Failed to count packages: {e}")
        
        # Calculate health score
        outdated_count = len(health_report['outdated_packages'])
        vulnerable_count = len(health_report['vulnerable_packages'])
        total_count = health_report['total_packages']
        
        if total_count > 0:
            # Penalize for outdated and vulnerable packages
            outdated_penalty = (outdated_count / total_count) * 3
            vulnerable_penalty = (vulnerable_count / total_count) * 7
            health_report['health_score'] = max(0, 10 - outdated_penalty - vulnerable_penalty)
        
        self.logger.info(f"Dependency health score: {health_report['health_score']:.1f}/10")
        if outdated_count > 0:
            self.logger.warning(f"Found {outdated_count} outdated packages")
        if vulnerable_count > 0:
            self.logger.warning(f"Found {vulnerable_count} vulnerable packages")
            
        return health_report
    
    def generate_maintenance_report(self) -> Dict[str, Any]:
        """Generate comprehensive maintenance report."""
        self.logger.info("Generating maintenance report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'repository_path': str(self.repo_path),
            'maintenance_tasks': {
                'cache_cleanup': 'completed',
                'log_cleanup': 'completed', 
                'temp_cleanup': 'completed',
                'git_optimization': 'completed' if not self.dry_run else 'skipped',
                'permission_validation': 'completed'
            },
            'health_checks': {
                'large_files': [],
                'dependency_health': {}
            },
            'recommendations': []
        }
        
        # Check for large files
        large_files = self.check_large_files()
        report['health_checks']['large_files'] = [str(f) for f in large_files]
        
        if large_files:
            report['recommendations'].append(
                "Consider using Git LFS for large files or removing them if unnecessary"
            )
        
        # Check dependency health
        dep_health = self.check_dependency_health()
        report['health_checks']['dependency_health'] = dep_health
        
        if dep_health['outdated_packages']:
            report['recommendations'].append(
                "Update outdated packages to latest versions"
            )
        
        if dep_health['vulnerable_packages']:
            report['recommendations'].append(
                "Address security vulnerabilities in dependencies immediately"
            )
        
        # Save report
        report_file = self.repo_path / f"maintenance-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Maintenance report saved to: {report_file}")
        return report
    
    def run_maintenance(self) -> None:
        """Run all maintenance tasks."""
        self.logger.info("Starting repository maintenance...")
        
        if self.dry_run:
            self.logger.info("Running in DRY RUN mode - no changes will be made")
        
        try:
            # Cleanup tasks
            self.cleanup_cache_files()
            self.cleanup_log_files()
            self.cleanup_temporary_files()
            
            # Optimization tasks
            if not self.dry_run:
                self.optimize_git_repository()
            
            # Validation tasks
            self.validate_file_permissions()
            
            # Generate report
            report = self.generate_maintenance_report()
            
            self.logger.info("Repository maintenance completed successfully")
            
            # Print summary
            if report['recommendations']:
                self.logger.info("Recommendations:")
                for rec in report['recommendations']:
                    self.logger.info(f"  - {rec}")
            else:
                self.logger.info("No maintenance recommendations")
                
        except Exception as e:
            self.logger.error(f"Maintenance failed: {e}")
            raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Repository maintenance script")
    parser.add_argument('--dry-run', action='store_true',
                      help='Run in dry-run mode (no actual changes)')
    parser.add_argument('--repo-path', type=Path, 
                      default=Path(__file__).parent.parent.parent,
                      help='Path to repository root')
    
    args = parser.parse_args()
    
    maintenance = RepositoryMaintenance(args.repo_path, dry_run=args.dry_run)
    maintenance.run_maintenance()


if __name__ == '__main__':
    main()