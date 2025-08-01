# Acousto-Gen Documentation

This directory contains comprehensive documentation for the Acousto-Gen toolkit.

## Documentation Structure

- **`api/`** - Auto-generated API documentation from docstrings
- **`user-guide/`** - User guides and tutorials
- **`examples/`** - Example notebooks and scripts
- **`architecture/`** - Technical architecture documentation
- **`deployment/`** - Deployment and production guides

## Building Documentation

To build the documentation locally:

```bash
# Install documentation dependencies
pip install -e .[dev]

# Build documentation
make docs

# Serve documentation locally
make docs-serve
```

The documentation will be available at http://localhost:8000

## Documentation Standards

- Use [Sphinx](https://www.sphinx-doc.org/) for documentation generation
- Follow [NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html)
- Include practical examples for all features
- Maintain clear architecture diagrams
- Keep deployment guides up-to-date

## Contributing to Documentation

1. Update docstrings in source code for API changes
2. Add examples for new features in `examples/`
3. Update user guides for workflow changes
4. Review documentation builds before submitting PRs

For more details, see our [Contributing Guide](../CONTRIBUTING.md).