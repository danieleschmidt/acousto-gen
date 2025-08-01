# CI/CD Workflow Documentation

This document outlines the recommended GitHub Actions workflows for the Acousto-Gen project.

## Required Workflows

### 1. Test and Quality Workflow
**File**: `.github/workflows/test.yml`

```yaml
name: Test and Quality
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Run pre-commit
      run: pre-commit run --all-files
    
    - name: Run tests
      run: |
        pytest --cov=acousto_gen --cov-report=xml --cov-report=term-missing
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### 2. Security Scanning Workflow
**File**: `.github/workflows/security.yml`

```yaml
name: Security Scanning
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 2 AM

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Run Bandit security scan
      run: bandit -r acousto_gen/ -f json -o bandit-report.json
    
    - name: Run Safety dependency scan
      run: safety check --json --output safety-report.json
    
    - name: Upload reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
```

### 3. Build and Release Workflow
**File**: `.github/workflows/release.yml`

```yaml
name: Build and Release
on:
  release:
    types: [published]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Check package
      run: twine check dist/*
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

### 4. Documentation Workflow
**File**: `.github/workflows/docs.yml`

```yaml
name: Documentation
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Build documentation
      run: |
        cd docs
        make html
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

## Dependabot Configuration

**File**: `.github/dependabot.yml`

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 10
    groups:
      security-updates:
        applies-to: security-updates
        patterns:
          - "*"
      minor-updates:
        update-types:
          - "minor"
          - "patch"
        patterns:
          - "*"
    
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
```

## Required Secrets

Configure these secrets in your GitHub repository settings:

- `PYPI_API_TOKEN`: For publishing to PyPI
- `CODECOV_TOKEN`: For code coverage reporting (optional)

## Branch Protection Rules

Recommended branch protection for `main`:

- Require pull request reviews before merging
- Require status checks to pass before merging:
  - `test (3.9)`
  - `test (3.10)` 
  - `test (3.11)`
  - `test (3.12)`
  - `security`
  - `docs`
- Require branches to be up to date before merging
- Restrict pushes that create files larger than 100MB

## Implementation Notes

1. **Start with the test workflow** - This provides the foundation
2. **Add security scanning** - Critical for production readiness  
3. **Setup documentation** - Improves developer experience
4. **Implement release automation** - Last step for full automation

## Monitoring and Alerts

Consider setting up:
- Slack/Discord notifications for failed builds
- Email alerts for security vulnerabilities
- Dashboard monitoring for build success rates

## Performance Optimization

- Use caching for pip dependencies
- Consider matrix parallelization for faster tests
- Implement artifact sharing between jobs when needed