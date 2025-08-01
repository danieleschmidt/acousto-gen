# Security Policy

## Supported Versions

We actively support the following versions of Acousto-Gen with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in Acousto-Gen, please report it to us in a responsible manner.

### Reporting Process

1. **Do NOT open a public GitHub issue** for security vulnerabilities
2. **Email us directly** at: [security@acousto-gen.org](mailto:security@acousto-gen.org)
3. **Include the following information:**
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact assessment
   - Suggested fix (if you have one)
   - Your contact information

### Response Timeline

- **Initial Response**: Within 48 hours
- **Assessment**: Within 5 business days
- **Fix Timeline**: Critical issues within 7 days, others within 30 days
- **Public Disclosure**: After fix is available and users have had time to update

### Security Best Practices

When using Acousto-Gen in production:

#### Hardware Safety
- Always implement proper safety limits for acoustic pressure
- Use hardware emergency stops for medical applications
- Regularly calibrate transducer arrays
- Monitor temperature and power consumption

#### Software Security
- Keep dependencies updated using `pip install --upgrade acousto-gen`
- Use virtual environments to isolate installations
- Validate all input parameters and pressure field constraints
- Implement proper error handling for hardware failures

#### Data Protection
- Encrypt sensitive calibration data at rest
- Use secure communication protocols for hardware control
- Implement access controls for multi-user systems
- Log safety-critical operations for audit trails

### Vulnerability Types We Address

- **Hardware Control**: Unauthorized transducer activation
- **Input Validation**: Malformed acoustic field parameters  
- **Resource Exhaustion**: Memory/GPU exhaustion attacks
- **Dependency Vulnerabilities**: Third-party package issues
- **Data Exposure**: Calibration or patient data leaks
- **Code Injection**: Unsafe eval/exec usage in optimization

### Security Contact

For urgent security matters, contact:
- Email: security@acousto-gen.org
- GPG Key: [Link to public key]

We appreciate responsible disclosure and will acknowledge security researchers who help improve Acousto-Gen's security.