# Changelog

All notable changes to Acousto-Gen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Enhanced SDLC tooling and documentation
- Security policy and vulnerability reporting process
- Code of conduct for community guidelines
- Comprehensive project structure for future development

### Changed
- Improved development documentation
- Enhanced pre-commit hooks configuration

### Fixed
- Code quality and security scanning improvements

## [0.1.0] - 2025-01-15

### Added
- Initial release of Acousto-Gen toolkit
- Core `AcousticHologram` class for basic holography operations
- Command-line interface (`acousto-gen` command)
- Comprehensive README with usage examples
- Development setup with modern Python tooling:
  - ruff for linting and formatting
  - mypy for static type checking
  - pytest for testing
  - pre-commit hooks for quality assurance
  - bandit for security scanning
- Basic project structure and package configuration
- MIT license
- Contributing guidelines and development documentation

### Technical Details
- Python 3.9+ support
- Dependencies: numpy, scipy, torch, matplotlib, plotly, h5py, pydantic, typer
- Modular architecture prepared for future expansion
- Comprehensive type hints and docstrings

---

## Release Notes Format

Each release includes:
- **Added**: New features and capabilities
- **Changed**: Changes to existing functionality  
- **Deprecated**: Soon-to-be removed features
- **Removed**: Features removed in this release
- **Fixed**: Bug fixes and corrections
- **Security**: Vulnerability fixes and security improvements

## Version Numbering

- **Major** (X.0.0): Breaking changes or major new features
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, no new features

## Development Builds

Development builds use semantic versioning with pre-release identifiers:
- `0.2.0-alpha.1`: Alpha releases
- `0.2.0-beta.1`: Beta releases  
- `0.2.0-rc.1`: Release candidates