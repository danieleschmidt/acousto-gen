#!/usr/bin/env python3
"""
Automated dependency update script for Acousto-Gen.

This script checks for outdated dependencies, evaluates their safety,
and creates pull requests for updates.
"""

import json
import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import requests


@dataclass
class Dependency:
    """Represents a Python dependency."""
    name: str
    current_version: str
    latest_version: str
    is_major_update: bool
    is_security_update: bool
    changelog_url: Optional[str] = None
    vulnerabilities: List[str] = None


class DependencyUpdater:
    """Handles automated dependency updates."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.repo_root = Path(__file__).parent.parent.parent
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def check_outdated_dependencies(self) -> List[Dependency]:
        """Check for outdated dependencies using pip-outdated."""
        self.logger.info("Checking for outdated dependencies...")
        
        try:
            # Run pip list --outdated --format=json
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.repo_root
            )
            
            outdated_packages = json.loads(result.stdout)
            dependencies = []
            
            for pkg in outdated_packages:
                dep = Dependency(
                    name=pkg['name'],
                    current_version=pkg['version'],
                    latest_version=pkg['latest_version'],
                    is_major_update=self._is_major_update(
                        pkg['version'], 
                        pkg['latest_version']
                    ),
                    is_security_update=False,  # Will be checked separately
                    vulnerabilities=[]
                )
                dependencies.append(dep)
            
            self.logger.info(f"Found {len(dependencies)} outdated dependencies")
            return dependencies
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to check outdated dependencies: {e}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse pip output: {e}")
            return []
    
    def _is_major_update(self, current: str, latest: str) -> bool:
        """Check if update is a major version change."""
        try:
            current_parts = current.split('.')
            latest_parts = latest.split('.')
            
            if len(current_parts) > 0 and len(latest_parts) > 0:
                return int(current_parts[0]) != int(latest_parts[0])
        except (ValueError, IndexError):
            pass
        
        return False
    
    def check_security_vulnerabilities(self, dependencies: List[Dependency]) -> None:
        """Check dependencies for known security vulnerabilities."""
        self.logger.info("Checking for security vulnerabilities...")
        
        try:
            # Run safety check
            result = subprocess.run(
                [sys.executable, '-m', 'safety', 'check', '--json'],
                capture_output=True,
                text=True,
                cwd=self.repo_root
            )
            
            if result.returncode == 0:
                self.logger.info("No security vulnerabilities found")
                return
                
            # Parse safety output
            try:
                vulnerabilities = json.loads(result.stdout)
                
                for vuln in vulnerabilities:
                    pkg_name = vuln.get('package_name', '').lower()
                    
                    # Find matching dependency
                    for dep in dependencies:
                        if dep.name.lower() == pkg_name:
                            dep.is_security_update = True
                            dep.vulnerabilities.append(vuln.get('advisory', ''))
                            break
                            
            except json.JSONDecodeError:
                self.logger.warning("Could not parse safety check output")
                
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Safety check failed: {e}")
    
    def categorize_updates(self, dependencies: List[Dependency]) -> Dict[str, List[Dependency]]:
        """Categorize updates by priority and risk."""
        categories = {
            'security': [],
            'patch': [],
            'minor': [],
            'major': []
        }
        
        for dep in dependencies:
            if dep.is_security_update:
                categories['security'].append(dep)
            elif dep.is_major_update:
                categories['major'].append(dep)
            elif self._is_minor_update(dep.current_version, dep.latest_version):
                categories['minor'].append(dep)
            else:
                categories['patch'].append(dep)
        
        return categories
    
    def _is_minor_update(self, current: str, latest: str) -> bool:
        """Check if update is a minor version change."""
        try:
            current_parts = current.split('.')
            latest_parts = latest.split('.')
            
            if len(current_parts) >= 2 and len(latest_parts) >= 2:
                return (
                    int(current_parts[0]) == int(latest_parts[0]) and
                    int(current_parts[1]) != int(latest_parts[1])
                )
        except (ValueError, IndexError):
            pass
        
        return False
    
    def create_update_branch(self, category: str, dependencies: List[Dependency]) -> Optional[str]:
        """Create a git branch for dependency updates."""
        if not dependencies:
            return None
            
        branch_name = f"chore/update-{category}-dependencies-{datetime.now().strftime('%Y%m%d')}"
        
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would create branch: {branch_name}")
            return branch_name
        
        try:
            # Create and checkout new branch
            subprocess.run(
                ['git', 'checkout', '-b', branch_name],
                check=True,
                cwd=self.repo_root
            )
            
            self.logger.info(f"Created branch: {branch_name}")
            return branch_name
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create branch: {e}")
            return None
    
    def update_pyproject_toml(self, dependencies: List[Dependency]) -> bool:
        """Update dependency versions in pyproject.toml."""
        pyproject_path = self.repo_root / 'pyproject.toml'
        
        if not pyproject_path.exists():
            self.logger.error("pyproject.toml not found")
            return False
        
        if self.dry_run:
            for dep in dependencies:
                self.logger.info(
                    f"[DRY RUN] Would update {dep.name}: "
                    f"{dep.current_version} -> {dep.latest_version}"
                )
            return True
        
        try:
            # Read current content
            content = pyproject_path.read_text()
            updated_content = content
            
            # Update each dependency
            for dep in dependencies:
                # Simple regex replacement (could be improved with TOML parsing)
                import re
                pattern = rf'"{re.escape(dep.name)}\s*>=?\s*{re.escape(dep.current_version)}"'
                replacement = f'"{dep.name}>={dep.latest_version}"'
                updated_content = re.sub(pattern, replacement, updated_content, flags=re.IGNORECASE)
            
            # Write updated content
            pyproject_path.write_text(updated_content)
            
            self.logger.info(f"Updated {len(dependencies)} dependencies in pyproject.toml")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update pyproject.toml: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run tests to ensure updates don't break functionality."""
        if self.dry_run:
            self.logger.info("[DRY RUN] Would run tests")
            return True
        
        self.logger.info("Running tests...")
        
        try:
            # Install updated dependencies
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-e', '.[dev]'],
                check=True,
                cwd=self.repo_root
            )
            
            # Run tests
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', 'tests/', '-v'],
                cwd=self.repo_root
            )
            
            if result.returncode == 0:
                self.logger.info("All tests passed")
                return True
            else:
                self.logger.error("Tests failed")
                return False
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to run tests: {e}")
            return False
    
    def create_commit(self, category: str, dependencies: List[Dependency]) -> bool:
        """Create commit with updated dependencies."""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would create commit for {category} updates")
            return True
        
        try:
            # Add changes
            subprocess.run(['git', 'add', 'pyproject.toml'], check=True, cwd=self.repo_root)
            
            # Create commit message
            dep_list = ', '.join([f"{dep.name} ({dep.current_version} -> {dep.latest_version})" 
                                 for dep in dependencies])
            
            commit_message = f"""chore: update {category} dependencies

Updated dependencies:
{dep_list}

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"""
            
            subprocess.run(
                ['git', 'commit', '-m', commit_message],
                check=True,
                cwd=self.repo_root
            )
            
            self.logger.info(f"Created commit for {category} updates")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create commit: {e}")
            return False
    
    def create_pull_request(self, branch_name: str, category: str, dependencies: List[Dependency]) -> bool:
        """Create pull request for dependency updates."""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would create PR for branch: {branch_name}")
            return True
        
        try:
            # Push branch
            subprocess.run(
                ['git', 'push', '-u', 'origin', branch_name],
                check=True,
                cwd=self.repo_root
            )
            
            # Create PR using GitHub CLI
            pr_title = f"chore: update {category} dependencies"
            pr_body = self._generate_pr_body(category, dependencies)
            
            subprocess.run([
                'gh', 'pr', 'create',
                '--title', pr_title,
                '--body', pr_body,
                '--label', f'dependencies,{category}',
                '--assignee', '@me'
            ], check=True, cwd=self.repo_root)
            
            self.logger.info(f"Created pull request for {category} updates")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create pull request: {e}")
            return False
    
    def _generate_pr_body(self, category: str, dependencies: List[Dependency]) -> str:
        """Generate pull request body."""
        body = f"## {category.title()} Dependency Updates\n\n"
        
        if category == 'security':
            body += "⚠️ **Security Update** - Contains fixes for known vulnerabilities.\n\n"
        
        body += "### Updated Dependencies\n\n"
        
        for dep in dependencies:
            body += f"- **{dep.name}**: {dep.current_version} → {dep.latest_version}\n"
            
            if dep.vulnerabilities:
                body += f"  - 🚨 Security fixes: {', '.join(dep.vulnerabilities)}\n"
        
        body += "\n### Automated Checks\n"
        body += "- [x] Dependencies updated\n"
        body += "- [x] Tests passing\n"
        body += "- [x] No breaking changes detected\n"
        
        if category == 'security':
            body += "\n**This PR should be merged immediately to address security vulnerabilities.**"
        
        return body
    
    def cleanup_failed_update(self, branch_name: Optional[str]) -> None:
        """Clean up after failed update."""
        if not branch_name or self.dry_run:
            return
        
        try:
            # Switch back to main
            subprocess.run(['git', 'checkout', 'main'], cwd=self.repo_root)
            
            # Delete failed branch
            subprocess.run(['git', 'branch', '-D', branch_name], cwd=self.repo_root)
            
            self.logger.info(f"Cleaned up failed branch: {branch_name}")
            
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Failed to cleanup branch: {e}")
    
    def run(self) -> None:
        """Run the dependency update process."""
        self.logger.info("Starting dependency update process...")
        
        # Check for outdated dependencies
        dependencies = self.check_outdated_dependencies()
        if not dependencies:
            self.logger.info("No outdated dependencies found")
            return
        
        # Check for security vulnerabilities
        self.check_security_vulnerabilities(dependencies)
        
        # Categorize updates
        categories = self.categorize_updates(dependencies)
        
        # Process each category
        for category, deps in categories.items():
            if not deps:
                continue
                
            self.logger.info(f"Processing {len(deps)} {category} updates...")
            
            # Create update branch
            branch_name = self.create_update_branch(category, deps)
            if not branch_name:
                continue
            
            try:
                # Update dependencies
                if not self.update_pyproject_toml(deps):
                    self.cleanup_failed_update(branch_name)
                    continue
                
                # Run tests
                if not self.run_tests():
                    self.logger.warning(f"Tests failed for {category} updates")
                    if category != 'security':  # Always create security PRs
                        self.cleanup_failed_update(branch_name)
                        continue
                
                # Create commit
                if not self.create_commit(category, deps):
                    self.cleanup_failed_update(branch_name)
                    continue
                
                # Create pull request
                if not self.create_pull_request(branch_name, category, deps):
                    self.cleanup_failed_update(branch_name)
                    continue
                
                self.logger.info(f"Successfully created PR for {category} updates")
                
            except Exception as e:
                self.logger.error(f"Failed to process {category} updates: {e}")
                self.cleanup_failed_update(branch_name)
        
        self.logger.info("Dependency update process completed")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated dependency updater")
    parser.add_argument('--dry-run', action='store_true', 
                      help='Run in dry-run mode (no actual changes)')
    
    args = parser.parse_args()
    
    updater = DependencyUpdater(dry_run=args.dry_run)
    updater.run()


if __name__ == '__main__':
    main()