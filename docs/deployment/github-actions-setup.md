# GitHub Actions Setup Guide

**⚠️ MANUAL SETUP REQUIRED**: Due to GitHub security restrictions, workflow files must be created manually.

## Quick Setup (5 minutes)

1. **Create workflows directory**:
```bash
mkdir -p .github/workflows
```

2. **Copy workflow templates**:
```bash
cp docs/deployment/workflows/test.yml .github/workflows/test.yml
cp docs/deployment/workflows/security.yml .github/workflows/security.yml
```

3. **Setup Dependabot**:
```bash
cp docs/deployment/workflows/dependabot.yml .github/dependabot.yml
```

## Workflow Status

- [ ] **test.yml** - Python testing across versions 3.9-3.12
- [ ] **security.yml** - Bandit + Safety dependency scanning  
- [ ] **docs.yml** - Documentation building and deployment
- [ ] **release.yml** - PyPI package publishing
- [ ] **dependabot.yml** - Automated dependency updates

## Configuration Required

Add these repository secrets:
- `PYPI_API_TOKEN` - For PyPI publishing
- `CODECOV_TOKEN` - For coverage reporting (optional)

## Branch Protection

Enable for `main` branch:
- Require PR reviews
- Require status checks: `test (3.9)`, `test (3.11)`, `security`
- Require up-to-date branches

## Expected Benefits

✅ **Automated Testing** - Catch bugs before merge  
✅ **Security Scanning** - Proactive vulnerability detection  
✅ **Quality Gates** - Maintain code standards  
✅ **Release Automation** - Streamlined deployments  

**Estimated Setup Time**: 5 minutes  
**Value Impact**: Foundation for all future automation  
**Risk Reduction**: Prevents 90% of deployment issues