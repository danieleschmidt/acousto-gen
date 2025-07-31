# Contributing to Acousto-Gen

Thank you for considering contributing to Acousto-Gen! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/acousto-gen.git
   cd acousto-gen
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   make dev-install
   # or: pip install -e .[dev]
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Development Workflow

### Code Quality

- **Formatting**: Use `ruff format` for code formatting
- **Linting**: Use `ruff check` for linting
- **Type checking**: Use `mypy` for static type checking
- **Pre-commit**: All commits are automatically checked

Run quality checks:
```bash
make quality  # Runs lint + type-check
```

### Testing

- Write tests for all new functionality
- Maintain high test coverage (>90%)
- Use descriptive test names and docstrings

Run tests:
```bash
make test          # Basic test run
make test-cov      # With coverage report
```

### Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Write tests** for your changes
3. **Ensure all checks pass**:
   ```bash
   make quality
   make test
   ```
4. **Update documentation** if needed
5. **Submit a pull request** with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots/examples if applicable

### Commit Messages

Follow conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/changes
- `refactor:` for code refactoring

Example: `feat: add multi-frequency optimization support`

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write descriptive docstrings (NumPy style)
- Keep line length ≤ 88 characters
- Use meaningful variable and function names

## Getting Help

- Check existing [issues](https://github.com/yourusername/acousto-gen/issues)
- Start a [discussion](https://github.com/yourusername/acousto-gen/discussions)
- Join our community chat (link to be added)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.