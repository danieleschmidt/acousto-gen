# Manual Setup Required

## ⚠️ Important: GitHub Workflow Setup

Due to GitHub App permission limitations, the following CI/CD workflows must be manually created by repository maintainers with appropriate permissions.

## Required Actions

### 1. Create Workflow Files

Copy the workflow templates from `docs/workflows/examples/` to `.github/workflows/`:

```bash
mkdir -p .github/workflows
cp docs/workflows/examples/*.yml .github/workflows/
```

### 2. Configure Repository Secrets

Add these secrets in GitHub repository settings (Settings → Secrets and variables → Actions):

#### Required Secrets
```
CODECOV_TOKEN           # Code coverage reporting
DOCKER_USERNAME         # Docker Hub username  
DOCKER_PASSWORD         # Docker Hub token
SLACK_WEBHOOK          # Slack notifications
SAFETY_WEBHOOK         # Critical safety alerts
```

#### Optional Secrets
```
NPM_TOKEN              # NPM publishing (if needed)
SENTRY_DSN             # Error tracking
DATADOG_API_KEY        # Performance monitoring
AWS_ACCESS_KEY_ID      # Cloud deployment
AWS_SECRET_ACCESS_KEY  # Cloud deployment
```

### 3. Configure Branch Protection

Enable branch protection for `main` branch:

1. Go to Settings → Branches
2. Add rule for `main` branch
3. Configure:
   - ✅ Require a pull request before merging
   - ✅ Require approvals (2)
   - ✅ Dismiss stale PR approvals when new commits are pushed
   - ✅ Require review from code owners
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Restrict pushes that create files
   - ✅ Required status checks:
     - `CI / Code Quality`
     - `CI / Security Scan`
     - `CI / Tests`
     - `CI / Docker Build`

### 4. Enable Security Features

Go to Settings → Security and enable:
- ✅ Dependency graph
- ✅ Dependabot alerts
- ✅ Dependabot security updates
- ✅ Secret scanning
- ✅ Code scanning alerts (CodeQL)

### 5. Configure Environments

Create deployment environments (Settings → Environments):

#### Staging Environment
- **Protection rules**: Require approval from maintainers
- **Environment secrets**: Staging deployment credentials
- **Environment variables**: `DEPLOYMENT_ENV=staging`

#### Production Environment
- **Protection rules**: Require approval from 2+ maintainers
- **Environment secrets**: Production deployment credentials  
- **Environment variables**: `DEPLOYMENT_ENV=production`

### 6. Required Permissions

The GitHub App needs these permissions to complete setup:
- **Actions**: Write (to create workflow files)
- **Administration**: Write (to configure branch protection)
- **Contents**: Write (to commit workflow files)
- **Metadata**: Read (to access repository information)
- **Pull requests**: Write (to manage PR requirements)
- **Security events**: Write (to manage security scanning)

## Validation

After setup, verify workflows are working:

```bash
# Create a test PR to trigger CI
git checkout -b test-workflows
echo "# Test" >> README.md
git add README.md
git commit -m "test: trigger CI workflows"
git push origin test-workflows

# Create PR and check that all status checks run
```

## Troubleshooting

### Workflow Not Triggering
- Check workflow file syntax: `yamllint .github/workflows/*.yml`
- Verify file permissions and location
- Check GitHub Actions tab for error messages

### Status Checks Not Required
- Ensure branch protection rules are correctly configured
- Check that workflow job names match required status checks
- Verify workflows have run at least once

### Secret Access Issues
- Confirm secrets are added at repository level
- Check secret names match workflow references exactly
- Verify organization settings don't block secret access

## Support

For additional help with workflow setup:
- 📧 Email: devops@acousto-gen.org
- 💬 Discord: #ci-cd-support
- 📚 Documentation: [CI/CD Guide](README.md)

## Security Note

⚠️ **Never commit secrets to version control**. All sensitive information must be stored in GitHub Secrets or environment variables.