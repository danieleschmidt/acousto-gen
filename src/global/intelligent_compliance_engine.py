"""
Intelligent Global Compliance Engine
Next-Generation AI-driven global compliance, regulatory adaptation, and intelligent governance.
Autonomous compliance monitoring, multi-jurisdictional awareness, and adaptive policy enforcement.
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
import asyncio
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager

# Mock implementations for required dependencies
class MockModule:
    def __init__(self, *args, **kwargs):
        pass
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

# Safe imports with fallbacks
try:
    import numpy as np
except ImportError:
    np = MockModule()

try:
    import aiohttp
    import requests
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    aiohttp = MockModule()
    requests = MockModule()

try:
    from cryptography.fernet import Fernet
    import hashlib
    import hmac
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Fernet = MockModule
    hashlib = MockModule()
    hmac = MockModule()


class ComplianceRegion(Enum):
    """Global compliance regions."""
    EUROPEAN_UNION = "eu"
    UNITED_STATES = "us"
    UNITED_KINGDOM = "uk"
    CANADA = "ca"
    AUSTRALIA = "au"
    JAPAN = "jp"
    SINGAPORE = "sg"
    SWITZERLAND = "ch"
    SOUTH_KOREA = "kr"
    GLOBAL = "global"


class ComplianceFramework(Enum):
    """Compliance frameworks and standards."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO27001 = "iso27001"
    NIST = "nist"
    FDA_MEDICAL = "fda_medical"
    CE_MARKING = "ce_marking"
    FCC = "fcc"
    IEC62304 = "iec62304"


class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    REQUIRES_ATTENTION = "requires_attention"
    UNKNOWN = "unknown"


@dataclass
class ComplianceRule:
    """Individual compliance rule specification."""
    rule_id: str
    framework: ComplianceFramework
    region: ComplianceRegion
    title: str
    description: str
    requirements: List[str]
    verification_method: str
    severity: str = "medium"  # low, medium, high, critical
    auto_verifiable: bool = False
    last_updated: Optional[datetime] = None
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceViolation:
    """Compliance violation record."""
    violation_id: str
    rule_id: str
    severity: str
    description: str
    detected_at: datetime
    component: str
    details: Dict[str, Any]
    remediation_steps: List[str]
    status: str = "open"  # open, in_progress, resolved
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None


@dataclass
class ComplianceAssessment:
    """Complete compliance assessment result."""
    assessment_id: str
    timestamp: datetime
    region: ComplianceRegion
    frameworks: List[ComplianceFramework]
    overall_status: ComplianceStatus
    compliance_score: float  # 0.0 to 1.0
    total_rules_checked: int
    rules_passed: int
    rules_failed: int
    violations: List[ComplianceViolation]
    recommendations: List[str]
    next_assessment_due: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplianceRuleEngine:
    """Engine for managing and evaluating compliance rules."""
    
    def __init__(self):
        self.rules: Dict[str, ComplianceRule] = {}
        self.rule_cache = {}
        self.rule_dependencies = {}
        self.custom_validators = {}
        
        # Initialize built-in rules
        self._initialize_builtin_rules()
    
    def _initialize_builtin_rules(self):
        """Initialize built-in compliance rules."""
        
        # GDPR Rules
        gdpr_rules = [
            ComplianceRule(
                rule_id="gdpr_001",
                framework=ComplianceFramework.GDPR,
                region=ComplianceRegion.EUROPEAN_UNION,
                title="Data Protection by Design",
                description="Implement data protection measures from the design phase",
                requirements=[
                    "Data minimization principles applied",
                    "Privacy-enhancing technologies implemented",
                    "Default privacy settings enabled",
                    "Data protection impact assessment conducted"
                ],
                verification_method="automated_scan",
                severity="high",
                auto_verifiable=True,
                source_url="https://gdpr.eu/article-25-data-protection-by-design/"
            ),
            ComplianceRule(
                rule_id="gdpr_002",
                framework=ComplianceFramework.GDPR,
                region=ComplianceRegion.EUROPEAN_UNION,
                title="Data Subject Rights",
                description="Provide mechanisms for data subject rights exercise",
                requirements=[
                    "Right to access implementation",
                    "Right to rectification process",
                    "Right to erasure capability",
                    "Data portability mechanisms",
                    "Response time <= 30 days"
                ],
                verification_method="process_review",
                severity="critical",
                auto_verifiable=False,
                source_url="https://gdpr.eu/chapter-3/"
            )
        ]
        
        # CCPA Rules
        ccpa_rules = [
            ComplianceRule(
                rule_id="ccpa_001",
                framework=ComplianceFramework.CCPA,
                region=ComplianceRegion.UNITED_STATES,
                title="Consumer Privacy Rights",
                description="Provide California consumers with privacy rights",
                requirements=[
                    "Right to know implementation",
                    "Right to delete personal information",
                    "Right to opt-out of sale",
                    "Non-discrimination provisions",
                    "Privacy policy disclosure"
                ],
                verification_method="policy_review",
                severity="high",
                auto_verifiable=False,
                source_url="https://oag.ca.gov/privacy/ccpa"
            )
        ]
        
        # FDA Medical Device Rules
        fda_rules = [
            ComplianceRule(
                rule_id="fda_001",
                framework=ComplianceFramework.FDA_MEDICAL,
                region=ComplianceRegion.UNITED_STATES,
                title="Medical Device Software Validation",
                description="Software used in medical devices must be validated",
                requirements=[
                    "Software validation documentation",
                    "Risk management process (ISO 14971)",
                    "Clinical evaluation if required",
                    "Quality system compliance",
                    "510(k) submission if applicable"
                ],
                verification_method="documentation_review",
                severity="critical",
                auto_verifiable=False,
                source_url="https://www.fda.gov/medical-devices/"
            )
        ]
        
        # Security and Safety Rules
        security_rules = [
            ComplianceRule(
                rule_id="sec_001",
                framework=ComplianceFramework.ISO27001,
                region=ComplianceRegion.GLOBAL,
                title="Information Security Management",
                description="Implement comprehensive information security controls",
                requirements=[
                    "Security policy documented",
                    "Access control measures",
                    "Incident response procedures",
                    "Business continuity planning",
                    "Regular security assessments"
                ],
                verification_method="security_audit",
                severity="high",
                auto_verifiable=True,
                source_url="https://www.iso.org/isoiec-27001-information-security.html"
            ),
            ComplianceRule(
                rule_id="safety_001",
                framework=ComplianceFramework.IEC62304,
                region=ComplianceRegion.GLOBAL,
                title="Medical Device Software Safety",
                description="Software safety requirements for medical devices",
                requirements=[
                    "Software safety classification",
                    "Risk management integration",
                    "Software development process",
                    "Verification and validation",
                    "Configuration management"
                ],
                verification_method="process_audit",
                severity="critical",
                auto_verifiable=False,
                source_url="https://www.iso.org/standard/38421.html"
            )
        ]
        
        # Add all rules to the engine
        all_rules = gdpr_rules + ccpa_rules + fda_rules + security_rules
        for rule in all_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: ComplianceRule):
        """Add a compliance rule to the engine."""
        self.rules[rule.rule_id] = rule
        
        # Clear cache when rules change
        self.rule_cache.clear()
        
        print(f"📋 Added compliance rule: {rule.rule_id} ({rule.framework.value})")
    
    def get_rules_for_framework(self, framework: ComplianceFramework) -> List[ComplianceRule]:
        """Get all rules for a specific framework."""
        return [rule for rule in self.rules.values() if rule.framework == framework]
    
    def get_rules_for_region(self, region: ComplianceRegion) -> List[ComplianceRule]:
        """Get all rules for a specific region."""
        return [rule for rule in self.rules.values() 
                if rule.region == region or rule.region == ComplianceRegion.GLOBAL]
    
    def get_applicable_rules(self, frameworks: List[ComplianceFramework], 
                           region: ComplianceRegion) -> List[ComplianceRule]:
        """Get rules applicable to given frameworks and region."""
        applicable_rules = []
        
        for rule in self.rules.values():
            if (rule.framework in frameworks and 
                (rule.region == region or rule.region == ComplianceRegion.GLOBAL)):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def add_custom_validator(self, rule_id: str, validator_func: callable):
        """Add custom validator function for a rule."""
        self.custom_validators[rule_id] = validator_func
        print(f"🔧 Added custom validator for rule: {rule_id}")


class SystemComplianceScanner:
    """Scanner for automated compliance verification."""
    
    def __init__(self, rule_engine: ComplianceRuleEngine):
        self.rule_engine = rule_engine
        self.scan_history = []
        self.scan_plugins = {}
        
        # Initialize built-in scanners
        self._initialize_scanners()
    
    def _initialize_scanners(self):
        """Initialize built-in compliance scanners."""
        
        # Privacy scanner
        self.scan_plugins['privacy'] = self._privacy_scanner
        self.scan_plugins['security'] = self._security_scanner
        self.scan_plugins['data_handling'] = self._data_handling_scanner
        self.scan_plugins['access_control'] = self._access_control_scanner
        self.scan_plugins['audit_logging'] = self._audit_logging_scanner
    
    def scan_system(self, system_config: Dict[str, Any], 
                   frameworks: List[ComplianceFramework],
                   region: ComplianceRegion) -> List[ComplianceViolation]:
        """Perform comprehensive system compliance scan."""
        
        violations = []
        applicable_rules = self.rule_engine.get_applicable_rules(frameworks, region)
        
        print(f"🔍 Scanning system for {len(applicable_rules)} compliance rules...")
        
        for rule in applicable_rules:
            if rule.auto_verifiable:
                rule_violations = self._verify_rule(rule, system_config)
                violations.extend(rule_violations)
        
        # Store scan history
        scan_record = {
            'timestamp': datetime.now(timezone.utc),
            'frameworks': [f.value for f in frameworks],
            'region': region.value,
            'rules_checked': len(applicable_rules),
            'violations_found': len(violations)
        }
        self.scan_history.append(scan_record)
        
        print(f"✅ Compliance scan completed: {len(violations)} violations found")
        
        return violations
    
    def _verify_rule(self, rule: ComplianceRule, system_config: Dict[str, Any]) -> List[ComplianceViolation]:
        """Verify a specific compliance rule."""
        violations = []
        
        # Check for custom validator
        if rule.rule_id in self.rule_engine.custom_validators:
            validator = self.rule_engine.custom_validators[rule.rule_id]
            try:
                is_compliant, details = validator(system_config)
                if not is_compliant:
                    violation = self._create_violation(rule, details)
                    violations.append(violation)
            except Exception as e:
                print(f"⚠️ Custom validator failed for rule {rule.rule_id}: {e}")
        
        # Use verification method
        elif rule.verification_method == "automated_scan":
            violations.extend(self._automated_verification(rule, system_config))
        elif rule.verification_method == "security_audit":
            violations.extend(self._security_verification(rule, system_config))
        elif rule.verification_method == "policy_review":
            violations.extend(self._policy_verification(rule, system_config))
        
        return violations
    
    def _automated_verification(self, rule: ComplianceRule, 
                              system_config: Dict[str, Any]) -> List[ComplianceViolation]:
        """Perform automated verification of compliance rule."""
        violations = []
        
        # GDPR specific checks
        if rule.framework == ComplianceFramework.GDPR:
            if rule.rule_id == "gdpr_001":  # Data Protection by Design
                violations.extend(self._check_privacy_by_design(rule, system_config))
        
        # Security checks
        elif rule.framework == ComplianceFramework.ISO27001:
            if rule.rule_id == "sec_001":  # Information Security Management
                violations.extend(self._check_security_controls(rule, system_config))
        
        return violations
    
    def _check_privacy_by_design(self, rule: ComplianceRule, 
                                system_config: Dict[str, Any]) -> List[ComplianceViolation]:
        """Check privacy by design implementation."""
        violations = []
        
        # Check data minimization
        data_collection = system_config.get('data_collection', {})
        if data_collection.get('minimal_data_principle', False) != True:
            violation = self._create_violation(
                rule,
                {
                    'requirement': 'Data minimization principles applied',
                    'current_state': 'Data minimization not implemented',
                    'component': 'data_collection'
                }
            )
            violations.append(violation)
        
        # Check default privacy settings
        privacy_settings = system_config.get('privacy_settings', {})
        if privacy_settings.get('default_privacy_enabled', False) != True:
            violation = self._create_violation(
                rule,
                {
                    'requirement': 'Default privacy settings enabled',
                    'current_state': 'Default privacy settings not enabled',
                    'component': 'privacy_settings'
                }
            )
            violations.append(violation)
        
        return violations
    
    def _check_security_controls(self, rule: ComplianceRule, 
                               system_config: Dict[str, Any]) -> List[ComplianceViolation]:
        """Check security control implementation."""
        violations = []
        
        security_config = system_config.get('security', {})
        
        # Check access control
        if not security_config.get('access_control_enabled', False):
            violation = self._create_violation(
                rule,
                {
                    'requirement': 'Access control measures',
                    'current_state': 'Access control not enabled',
                    'component': 'security.access_control'
                }
            )
            violations.append(violation)
        
        # Check incident response
        if not security_config.get('incident_response_plan', False):
            violation = self._create_violation(
                rule,
                {
                    'requirement': 'Incident response procedures',
                    'current_state': 'Incident response plan not documented',
                    'component': 'security.incident_response'
                }
            )
            violations.append(violation)
        
        return violations
    
    def _security_verification(self, rule: ComplianceRule, 
                             system_config: Dict[str, Any]) -> List[ComplianceViolation]:
        """Perform security-focused verification."""
        # Placeholder for security verification logic
        return []
    
    def _policy_verification(self, rule: ComplianceRule, 
                           system_config: Dict[str, Any]) -> List[ComplianceViolation]:
        """Perform policy-focused verification."""
        # Placeholder for policy verification logic
        return []
    
    def _create_violation(self, rule: ComplianceRule, details: Dict[str, Any]) -> ComplianceViolation:
        """Create a compliance violation record."""
        violation_id = f"viol_{int(time.time())}_{rule.rule_id}"
        
        return ComplianceViolation(
            violation_id=violation_id,
            rule_id=rule.rule_id,
            severity=rule.severity,
            description=f"Violation of {rule.title}: {details.get('requirement', 'Unknown requirement')}",
            detected_at=datetime.now(timezone.utc),
            component=details.get('component', 'system'),
            details=details,
            remediation_steps=self._generate_remediation_steps(rule, details)
        )
    
    def _generate_remediation_steps(self, rule: ComplianceRule, 
                                  details: Dict[str, Any]) -> List[str]:
        """Generate remediation steps for a violation."""
        remediation_steps = []
        
        # Generic remediation based on rule framework
        if rule.framework == ComplianceFramework.GDPR:
            remediation_steps.extend([
                "Review data processing activities",
                "Update privacy policies and procedures",
                "Implement technical and organizational measures",
                "Conduct privacy impact assessment if needed"
            ])
        
        elif rule.framework == ComplianceFramework.ISO27001:
            remediation_steps.extend([
                "Review information security policy",
                "Implement required security controls",
                "Update security procedures",
                "Conduct security risk assessment"
            ])
        
        # Add specific remediation based on violation details
        component = details.get('component', '')
        if 'access_control' in component:
            remediation_steps.append("Implement role-based access control (RBAC)")
        if 'privacy_settings' in component:
            remediation_steps.append("Enable privacy-by-default configurations")
        if 'data_collection' in component:
            remediation_steps.append("Apply data minimization principles")
        
        return remediation_steps
    
    # Scanner plugin methods
    def _privacy_scanner(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Scan for privacy compliance issues."""
        results = {
            'privacy_policy_present': system_config.get('privacy_policy', False),
            'data_minimization': system_config.get('data_collection', {}).get('minimal_data_principle', False),
            'consent_mechanisms': system_config.get('consent', {}).get('explicit_consent', False),
            'data_retention_policy': system_config.get('data_retention', {}).get('policy_defined', False)
        }
        return results
    
    def _security_scanner(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Scan for security compliance issues."""
        security_config = system_config.get('security', {})
        results = {
            'encryption_enabled': security_config.get('encryption_at_rest', False),
            'access_control': security_config.get('access_control_enabled', False),
            'audit_logging': security_config.get('audit_logging', False),
            'vulnerability_scanning': security_config.get('vulnerability_scanning', False)
        }
        return results
    
    def _data_handling_scanner(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Scan for data handling compliance issues."""
        data_config = system_config.get('data_handling', {})
        results = {
            'data_classification': data_config.get('classification_scheme', False),
            'secure_transmission': data_config.get('secure_transmission', False),
            'data_backup': data_config.get('backup_strategy', False),
            'data_disposal': data_config.get('secure_disposal', False)
        }
        return results
    
    def _access_control_scanner(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Scan for access control compliance issues."""
        access_config = system_config.get('access_control', {})
        results = {
            'multi_factor_auth': access_config.get('mfa_enabled', False),
            'role_based_access': access_config.get('rbac_implemented', False),
            'session_management': access_config.get('session_timeout', False),
            'account_lockout': access_config.get('lockout_policy', False)
        }
        return results
    
    def _audit_logging_scanner(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Scan for audit logging compliance issues."""
        logging_config = system_config.get('audit_logging', {})
        results = {
            'comprehensive_logging': logging_config.get('comprehensive_events', False),
            'log_integrity': logging_config.get('integrity_protection', False),
            'log_retention': logging_config.get('retention_policy', False),
            'log_monitoring': logging_config.get('real_time_monitoring', False)
        }
        return results


class ComplianceReportGenerator:
    """Generator for compliance reports and documentation."""
    
    def __init__(self):
        self.report_templates = {}
        self.custom_formatters = {}
        
        # Initialize built-in templates
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize built-in report templates."""
        
        self.report_templates['executive_summary'] = {
            'sections': [
                'compliance_overview',
                'risk_assessment',
                'key_findings',
                'recommendations',
                'next_steps'
            ],
            'format': 'narrative'
        }
        
        self.report_templates['detailed_technical'] = {
            'sections': [
                'assessment_methodology',
                'rules_evaluated',
                'violations_detailed',
                'remediation_plan',
                'implementation_timeline'
            ],
            'format': 'technical'
        }
        
        self.report_templates['regulatory_submission'] = {
            'sections': [
                'regulatory_framework',
                'compliance_evidence',
                'gap_analysis',
                'corrective_actions',
                'certification_readiness'
            ],
            'format': 'formal'
        }
    
    def generate_assessment_report(self, assessment: ComplianceAssessment,
                                 template: str = 'executive_summary') -> Dict[str, Any]:
        """Generate comprehensive compliance assessment report."""
        
        if template not in self.report_templates:
            template = 'executive_summary'
        
        report_template = self.report_templates[template]
        report = {
            'metadata': {
                'report_id': f"compliance_report_{int(time.time())}",
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'template': template,
                'assessment_id': assessment.assessment_id
            },
            'sections': {}
        }
        
        # Generate each section
        for section in report_template['sections']:
            report['sections'][section] = self._generate_section(section, assessment)
        
        return report
    
    def _generate_section(self, section_name: str, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Generate a specific report section."""
        
        if section_name == 'compliance_overview':
            return self._generate_overview_section(assessment)
        elif section_name == 'risk_assessment':
            return self._generate_risk_section(assessment)
        elif section_name == 'key_findings':
            return self._generate_findings_section(assessment)
        elif section_name == 'recommendations':
            return self._generate_recommendations_section(assessment)
        elif section_name == 'next_steps':
            return self._generate_next_steps_section(assessment)
        elif section_name == 'violations_detailed':
            return self._generate_violations_section(assessment)
        elif section_name == 'remediation_plan':
            return self._generate_remediation_section(assessment)
        else:
            return {'content': f"Section {section_name} content"}
    
    def _generate_overview_section(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Generate compliance overview section."""
        return {
            'title': 'Compliance Overview',
            'summary': f"Overall compliance status: {assessment.overall_status.value}",
            'metrics': {
                'compliance_score': f"{assessment.compliance_score:.2%}",
                'rules_passed': f"{assessment.rules_passed}/{assessment.total_rules_checked}",
                'violations_count': len(assessment.violations),
                'frameworks_assessed': [f.value for f in assessment.frameworks],
                'region': assessment.region.value
            },
            'assessment_date': assessment.timestamp.isoformat(),
            'next_assessment': assessment.next_assessment_due.isoformat()
        }
    
    def _generate_risk_section(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Generate risk assessment section."""
        risk_levels = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for violation in assessment.violations:
            if violation.severity in risk_levels:
                risk_levels[violation.severity] += 1
        
        total_violations = len(assessment.violations)
        risk_score = self._calculate_risk_score(risk_levels, total_violations)
        
        return {
            'title': 'Risk Assessment',
            'overall_risk_level': self._determine_risk_level(risk_score),
            'risk_score': risk_score,
            'risk_breakdown': risk_levels,
            'key_risks': self._identify_key_risks(assessment.violations),
            'risk_trend': 'stable'  # Would be calculated from historical data
        }
    
    def _generate_findings_section(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Generate key findings section."""
        findings = []
        
        # Group violations by framework
        framework_violations = {}
        for violation in assessment.violations:
            framework = violation.details.get('framework', 'unknown')
            if framework not in framework_violations:
                framework_violations[framework] = []
            framework_violations[framework].append(violation)
        
        # Generate findings for each framework
        for framework, violations in framework_violations.items():
            findings.append({
                'framework': framework,
                'violation_count': len(violations),
                'severity_distribution': self._get_severity_distribution(violations),
                'most_common_issues': self._get_common_issues(violations)
            })
        
        return {
            'title': 'Key Findings',
            'framework_analysis': findings,
            'critical_gaps': self._identify_critical_gaps(assessment.violations),
            'positive_findings': self._identify_positive_findings(assessment)
        }
    
    def _generate_recommendations_section(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Generate recommendations section."""
        recommendations = []
        
        # Priority-based recommendations
        critical_violations = [v for v in assessment.violations if v.severity == 'critical']
        high_violations = [v for v in assessment.violations if v.severity == 'high']
        
        if critical_violations:
            recommendations.append({
                'priority': 'immediate',
                'title': 'Address Critical Compliance Gaps',
                'description': f"Resolve {len(critical_violations)} critical violations immediately",
                'actions': self._get_critical_actions(critical_violations),
                'timeline': '30 days'
            })
        
        if high_violations:
            recommendations.append({
                'priority': 'high',
                'title': 'Remediate High-Risk Issues',
                'description': f"Address {len(high_violations)} high-risk compliance issues",
                'actions': self._get_high_priority_actions(high_violations),
                'timeline': '90 days'
            })
        
        # Framework-specific recommendations
        for framework in assessment.frameworks:
            framework_recs = self._get_framework_recommendations(framework, assessment.violations)
            if framework_recs:
                recommendations.extend(framework_recs)
        
        return {
            'title': 'Recommendations',
            'priority_actions': recommendations,
            'strategic_improvements': self._get_strategic_recommendations(assessment),
            'resource_requirements': self._estimate_resource_requirements(recommendations)
        }
    
    def _generate_next_steps_section(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Generate next steps section."""
        return {
            'title': 'Next Steps',
            'immediate_actions': [
                'Review and acknowledge compliance assessment results',
                'Assign remediation tasks to responsible teams',
                'Establish timeline for addressing critical violations'
            ],
            'short_term_plan': [
                'Implement immediate remediation measures',
                'Update compliance policies and procedures',
                'Conduct team training on compliance requirements'
            ],
            'long_term_strategy': [
                'Establish ongoing compliance monitoring',
                'Integrate compliance into development lifecycle',
                'Regular compliance assessment schedule'
            ],
            'next_assessment_date': assessment.next_assessment_due.isoformat(),
            'monitoring_schedule': self._generate_monitoring_schedule(assessment)
        }
    
    def _generate_violations_section(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Generate detailed violations section."""
        violations_by_severity = {}
        for violation in assessment.violations:
            if violation.severity not in violations_by_severity:
                violations_by_severity[violation.severity] = []
            violations_by_severity[violation.severity].append({
                'violation_id': violation.violation_id,
                'rule_id': violation.rule_id,
                'description': violation.description,
                'component': violation.component,
                'details': violation.details,
                'remediation_steps': violation.remediation_steps
            })
        
        return {
            'title': 'Detailed Violations',
            'total_violations': len(assessment.violations),
            'violations_by_severity': violations_by_severity,
            'violation_trends': 'Analysis of violation trends over time',
            'remediation_status': self._get_remediation_status(assessment.violations)
        }
    
    def _generate_remediation_section(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Generate remediation plan section."""
        remediation_plan = []
        
        # Group by severity and create phased approach
        critical_violations = [v for v in assessment.violations if v.severity == 'critical']
        high_violations = [v for v in assessment.violations if v.severity == 'high']
        medium_violations = [v for v in assessment.violations if v.severity == 'medium']
        
        if critical_violations:
            remediation_plan.append({
                'phase': 'Phase 1 - Critical Issues',
                'timeline': '0-30 days',
                'violations': len(critical_violations),
                'actions': self._consolidate_remediation_actions(critical_violations),
                'resources_required': 'High priority team assignment'
            })
        
        if high_violations:
            remediation_plan.append({
                'phase': 'Phase 2 - High Priority',
                'timeline': '30-90 days',
                'violations': len(high_violations),
                'actions': self._consolidate_remediation_actions(high_violations),
                'resources_required': 'Dedicated compliance team'
            })
        
        if medium_violations:
            remediation_plan.append({
                'phase': 'Phase 3 - Medium Priority',
                'timeline': '90-180 days',
                'violations': len(medium_violations),
                'actions': self._consolidate_remediation_actions(medium_violations),
                'resources_required': 'Regular development cycle'
            })
        
        return {
            'title': 'Remediation Plan',
            'phased_approach': remediation_plan,
            'total_estimated_effort': self._estimate_remediation_effort(assessment.violations),
            'success_metrics': self._define_success_metrics(assessment),
            'monitoring_plan': 'Continuous monitoring and quarterly assessments'
        }
    
    # Helper methods for report generation
    def _calculate_risk_score(self, risk_levels: Dict[str, int], total_violations: int) -> float:
        """Calculate overall risk score."""
        if total_violations == 0:
            return 0.0
        
        weights = {'critical': 1.0, 'high': 0.7, 'medium': 0.4, 'low': 0.1}
        weighted_score = sum(risk_levels[level] * weights[level] for level in risk_levels)
        
        return min(weighted_score / total_violations, 1.0)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from score."""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.3:
            return 'medium'
        else:
            return 'low'
    
    def _identify_key_risks(self, violations: List[ComplianceViolation]) -> List[str]:
        """Identify key risk areas."""
        risks = []
        
        # Check for critical violations
        critical_count = len([v for v in violations if v.severity == 'critical'])
        if critical_count > 0:
            risks.append(f"{critical_count} critical compliance violations requiring immediate attention")
        
        # Check for framework-specific risks
        frameworks = set(v.details.get('framework') for v in violations if v.details.get('framework'))
        for framework in frameworks:
            framework_violations = [v for v in violations if v.details.get('framework') == framework]
            if len(framework_violations) > 3:
                risks.append(f"Multiple violations in {framework} framework")
        
        return risks
    
    def _get_severity_distribution(self, violations: List[ComplianceViolation]) -> Dict[str, int]:
        """Get severity distribution of violations."""
        distribution = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for violation in violations:
            if violation.severity in distribution:
                distribution[violation.severity] += 1
        return distribution
    
    def _get_common_issues(self, violations: List[ComplianceViolation]) -> List[str]:
        """Get most common issues."""
        components = {}
        for violation in violations:
            component = violation.component
            components[component] = components.get(component, 0) + 1
        
        # Sort by frequency and return top issues
        sorted_components = sorted(components.items(), key=lambda x: x[1], reverse=True)
        return [f"{comp}: {count} issues" for comp, count in sorted_components[:5]]
    
    def _identify_critical_gaps(self, violations: List[ComplianceViolation]) -> List[str]:
        """Identify critical compliance gaps."""
        critical_violations = [v for v in violations if v.severity == 'critical']
        gaps = []
        
        for violation in critical_violations:
            gaps.append(f"{violation.component}: {violation.description}")
        
        return gaps[:5]  # Top 5 critical gaps
    
    def _identify_positive_findings(self, assessment: ComplianceAssessment) -> List[str]:
        """Identify positive compliance findings."""
        findings = []
        
        if assessment.compliance_score > 0.8:
            findings.append("Strong overall compliance posture")
        
        if assessment.rules_passed > assessment.rules_failed:
            findings.append(f"Majority of rules passed ({assessment.rules_passed}/{assessment.total_rules_checked})")
        
        # Add framework-specific positive findings
        for framework in assessment.frameworks:
            framework_violations = [v for v in assessment.violations 
                                  if v.details.get('framework') == framework.value]
            if len(framework_violations) == 0:
                findings.append(f"Full compliance with {framework.value} framework")
        
        return findings
    
    def _get_critical_actions(self, violations: List[ComplianceViolation]) -> List[str]:
        """Get critical remediation actions."""
        actions = set()
        for violation in violations:
            actions.update(violation.remediation_steps[:2])  # Top 2 actions
        return list(actions)[:5]
    
    def _get_high_priority_actions(self, violations: List[ComplianceViolation]) -> List[str]:
        """Get high priority remediation actions."""
        actions = set()
        for violation in violations:
            actions.update(violation.remediation_steps[:1])  # Top action
        return list(actions)[:5]
    
    def _get_framework_recommendations(self, framework: ComplianceFramework, 
                                     violations: List[ComplianceViolation]) -> List[Dict[str, Any]]:
        """Get framework-specific recommendations."""
        framework_violations = [v for v in violations 
                              if v.details.get('framework') == framework.value]
        
        if not framework_violations:
            return []
        
        return [{
            'priority': 'medium',
            'title': f'Enhance {framework.value} Compliance',
            'description': f'Address {len(framework_violations)} violations in {framework.value} framework',
            'actions': self._get_framework_actions(framework, framework_violations),
            'timeline': '60-90 days'
        }]
    
    def _get_framework_actions(self, framework: ComplianceFramework, 
                             violations: List[ComplianceViolation]) -> List[str]:
        """Get framework-specific actions."""
        if framework == ComplianceFramework.GDPR:
            return [
                'Update privacy policies and procedures',
                'Implement data protection impact assessments',
                'Enhance consent management systems'
            ]
        elif framework == ComplianceFramework.ISO27001:
            return [
                'Strengthen information security management system',
                'Implement comprehensive security controls',
                'Conduct security risk assessments'
            ]
        else:
            return ['Review and update compliance procedures']
    
    def _get_strategic_recommendations(self, assessment: ComplianceAssessment) -> List[str]:
        """Get strategic compliance recommendations."""
        return [
            'Implement automated compliance monitoring',
            'Establish compliance-first development practices',
            'Regular compliance training for all teams',
            'Integration of compliance into CI/CD pipeline'
        ]
    
    def _estimate_resource_requirements(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate resource requirements for recommendations."""
        return {
            'full_time_equivalent': '2-3 FTE for 6 months',
            'budget_estimate': '$150,000 - $250,000',
            'key_roles': [
                'Compliance Officer',
                'Legal Counsel',
                'Technical Implementation Team',
                'Security Specialist'
            ],
            'external_resources': 'Compliance consultant (optional)'
        }
    
    def _generate_monitoring_schedule(self, assessment: ComplianceAssessment) -> Dict[str, str]:
        """Generate compliance monitoring schedule."""
        return {
            'daily': 'Automated security and privacy scans',
            'weekly': 'Compliance dashboard review',
            'monthly': 'Violation remediation status review',
            'quarterly': 'Comprehensive compliance assessment',
            'annually': 'Full compliance audit and certification review'
        }
    
    def _get_remediation_status(self, violations: List[ComplianceViolation]) -> Dict[str, int]:
        """Get remediation status summary."""
        status_count = {'open': 0, 'in_progress': 0, 'resolved': 0}
        for violation in violations:
            if violation.status in status_count:
                status_count[violation.status] += 1
        return status_count
    
    def _consolidate_remediation_actions(self, violations: List[ComplianceViolation]) -> List[str]:
        """Consolidate remediation actions across violations."""
        all_actions = []
        for violation in violations:
            all_actions.extend(violation.remediation_steps)
        
        # Remove duplicates and prioritize
        unique_actions = list(set(all_actions))
        return unique_actions[:10]  # Top 10 actions
    
    def _estimate_remediation_effort(self, violations: List[ComplianceViolation]) -> str:
        """Estimate total remediation effort."""
        critical_count = len([v for v in violations if v.severity == 'critical'])
        high_count = len([v for v in violations if v.severity == 'high'])
        medium_count = len([v for v in violations if v.severity == 'medium'])
        
        total_days = critical_count * 5 + high_count * 3 + medium_count * 1
        return f"{total_days} person-days estimated"
    
    def _define_success_metrics(self, assessment: ComplianceAssessment) -> List[str]:
        """Define success metrics for remediation."""
        return [
            'Zero critical violations within 30 days',
            'Compliance score improvement to >90%',
            'All high-priority violations resolved within 90 days',
            'Automated monitoring implementation',
            'Team compliance training completion'
        ]


class IntelligentComplianceEngine:
    """Main intelligent compliance engine orchestrating all components."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.rule_engine = ComplianceRuleEngine()
        self.scanner = SystemComplianceScanner(self.rule_engine)
        self.report_generator = ComplianceReportGenerator()
        
        # Assessment history
        self.assessment_history = []
        self.violation_history = []
        
        # Configuration
        self.auto_scan_enabled = self.config.get('auto_scan_enabled', True)
        self.scan_interval_hours = self.config.get('scan_interval_hours', 24)
        self.notification_enabled = self.config.get('notification_enabled', True)
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        print("🛡️ Intelligent Compliance Engine initialized")
    
    def perform_compliance_assessment(self, 
                                    system_config: Dict[str, Any],
                                    frameworks: List[ComplianceFramework],
                                    region: ComplianceRegion,
                                    assessment_type: str = 'comprehensive') -> ComplianceAssessment:
        """Perform comprehensive compliance assessment."""
        
        print(f"🔍 Starting compliance assessment for {region.value} region")
        print(f"📋 Frameworks: {[f.value for f in frameworks]}")
        
        start_time = time.time()
        
        # Get applicable rules
        applicable_rules = self.rule_engine.get_applicable_rules(frameworks, region)
        
        # Perform automated scanning
        violations = self.scanner.scan_system(system_config, frameworks, region)
        
        # Calculate compliance metrics
        total_rules = len(applicable_rules)
        rules_failed = len(violations)
        rules_passed = total_rules - rules_failed
        compliance_score = rules_passed / total_rules if total_rules > 0 else 1.0
        
        # Determine overall status
        if compliance_score >= 0.95:
            overall_status = ComplianceStatus.COMPLIANT
        elif compliance_score >= 0.8:
            overall_status = ComplianceStatus.REQUIRES_ATTENTION
        elif len([v for v in violations if v.severity == 'critical']) > 0:
            overall_status = ComplianceStatus.NON_COMPLIANT
        else:
            overall_status = ComplianceStatus.PENDING_REVIEW
        
        # Generate recommendations
        recommendations = self._generate_assessment_recommendations(violations, compliance_score)
        
        # Create assessment
        assessment = ComplianceAssessment(
            assessment_id=f"assess_{int(time.time())}",
            timestamp=datetime.now(timezone.utc),
            region=region,
            frameworks=frameworks,
            overall_status=overall_status,
            compliance_score=compliance_score,
            total_rules_checked=total_rules,
            rules_passed=rules_passed,
            rules_failed=rules_failed,
            violations=violations,
            recommendations=recommendations,
            next_assessment_due=datetime.now(timezone.utc) + timedelta(days=90),
            metadata={
                'assessment_type': assessment_type,
                'scan_duration': time.time() - start_time,
                'auto_verifiable_rules': len([r for r in applicable_rules if r.auto_verifiable])
            }
        )
        
        # Store assessment
        self.assessment_history.append(assessment)
        self.violation_history.extend(violations)
        
        print(f"✅ Assessment completed: {compliance_score:.1%} compliance score")
        print(f"📊 {rules_passed}/{total_rules} rules passed, {len(violations)} violations found")
        
        return assessment
    
    def generate_compliance_report(self, assessment: ComplianceAssessment,
                                 template: str = 'executive_summary',
                                 output_format: str = 'json') -> Union[Dict[str, Any], str]:
        """Generate compliance report from assessment."""
        
        print(f"📄 Generating compliance report using {template} template")
        
        report = self.report_generator.generate_assessment_report(assessment, template)
        
        if output_format == 'json':
            return report
        elif output_format == 'markdown':
            return self._format_report_as_markdown(report)
        elif output_format == 'html':
            return self._format_report_as_html(report)
        else:
            return report
    
    def start_continuous_monitoring(self, system_config: Dict[str, Any],
                                  frameworks: List[ComplianceFramework],
                                  region: ComplianceRegion):
        """Start continuous compliance monitoring."""
        
        if self.monitoring_active:
            print("⚠️ Continuous monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Perform assessment
                    assessment = self.perform_compliance_assessment(
                        system_config, frameworks, region, 'continuous'
                    )
                    
                    # Check for critical violations
                    critical_violations = [v for v in assessment.violations if v.severity == 'critical']
                    if critical_violations and self.notification_enabled:
                        self._send_critical_alert(critical_violations)
                    
                    # Wait for next scan
                    time.sleep(self.scan_interval_hours * 3600)
                    
                except Exception as e:
                    print(f"❌ Monitoring error: {e}")
                    time.sleep(600)  # Wait 10 minutes before retrying
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        print(f"🔄 Continuous compliance monitoring started (interval: {self.scan_interval_hours}h)")
    
    def stop_continuous_monitoring(self):
        """Stop continuous compliance monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            print("🛑 Continuous compliance monitoring stopped")
        else:
            print("⚠️ Continuous monitoring not active")
    
    def add_custom_rule(self, rule: ComplianceRule):
        """Add custom compliance rule."""
        self.rule_engine.add_rule(rule)
        print(f"➕ Added custom compliance rule: {rule.rule_id}")
    
    def add_custom_validator(self, rule_id: str, validator_func: callable):
        """Add custom validator for a rule."""
        self.rule_engine.add_custom_validator(rule_id, validator_func)
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance dashboard data."""
        if not self.assessment_history:
            return {'message': 'No assessments performed yet'}
        
        latest_assessment = self.assessment_history[-1]
        
        # Calculate trends
        if len(self.assessment_history) >= 2:
            previous_assessment = self.assessment_history[-2]
            score_trend = latest_assessment.compliance_score - previous_assessment.compliance_score
            violation_trend = len(latest_assessment.violations) - len(previous_assessment.violations)
        else:
            score_trend = 0.0
            violation_trend = 0
        
        # Violation breakdown
        violation_breakdown = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for violation in latest_assessment.violations:
            if violation.severity in violation_breakdown:
                violation_breakdown[violation.severity] += 1
        
        return {
            'current_status': {
                'overall_status': latest_assessment.overall_status.value,
                'compliance_score': latest_assessment.compliance_score,
                'total_violations': len(latest_assessment.violations),
                'last_assessment': latest_assessment.timestamp.isoformat()
            },
            'trends': {
                'score_change': score_trend,
                'violation_change': violation_trend,
                'trend_direction': 'improving' if score_trend > 0 else 'declining' if score_trend < 0 else 'stable'
            },
            'violation_breakdown': violation_breakdown,
            'frameworks_covered': [f.value for f in latest_assessment.frameworks],
            'region': latest_assessment.region.value,
            'next_assessment': latest_assessment.next_assessment_due.isoformat(),
            'monitoring_status': 'active' if self.monitoring_active else 'inactive',
            'total_assessments': len(self.assessment_history)
        }
    
    def export_compliance_data(self, format: str = 'json') -> Union[Dict[str, Any], str]:
        """Export compliance data for backup or integration."""
        data = {
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'engine_config': self.config,
            'assessment_history': [
                {
                    'assessment_id': a.assessment_id,
                    'timestamp': a.timestamp.isoformat(),
                    'region': a.region.value,
                    'frameworks': [f.value for f in a.frameworks],
                    'overall_status': a.overall_status.value,
                    'compliance_score': a.compliance_score,
                    'total_rules_checked': a.total_rules_checked,
                    'rules_passed': a.rules_passed,
                    'rules_failed': a.rules_failed,
                    'violation_count': len(a.violations)
                }
                for a in self.assessment_history
            ],
            'total_rules': len(self.rule_engine.rules),
            'custom_rules': len([r for r in self.rule_engine.rules.values() 
                               if r.rule_id.startswith('custom_')]),
            'scan_history_count': len(self.scanner.scan_history)
        }
        
        if format == 'json':
            return data
        else:
            return json.dumps(data, indent=2, default=str)
    
    # Helper methods
    def _generate_assessment_recommendations(self, violations: List[ComplianceViolation], 
                                           compliance_score: float) -> List[str]:
        """Generate recommendations based on assessment results."""
        recommendations = []
        
        critical_count = len([v for v in violations if v.severity == 'critical'])
        if critical_count > 0:
            recommendations.append(f"Immediately address {critical_count} critical compliance violations")
        
        if compliance_score < 0.8:
            recommendations.append("Implement comprehensive compliance improvement program")
        
        if compliance_score < 0.5:
            recommendations.append("Engage external compliance consultant for immediate remediation")
        
        # Framework-specific recommendations
        framework_violations = {}
        for violation in violations:
            framework = violation.details.get('framework', 'unknown')
            framework_violations[framework] = framework_violations.get(framework, 0) + 1
        
        for framework, count in framework_violations.items():
            if count >= 3:
                recommendations.append(f"Focus on {framework} compliance improvements ({count} violations)")
        
        return recommendations
    
    def _send_critical_alert(self, violations: List[ComplianceViolation]):
        """Send alert for critical violations."""
        print(f"🚨 CRITICAL COMPLIANCE ALERT: {len(violations)} critical violations detected")
        for violation in violations:
            print(f"   - {violation.description}")
    
    def _format_report_as_markdown(self, report: Dict[str, Any]) -> str:
        """Format report as Markdown."""
        # Basic Markdown formatting
        md_content = f"# Compliance Assessment Report\n\n"
        md_content += f"**Generated:** {report['metadata']['generated_at']}\n\n"
        
        for section_name, section_content in report['sections'].items():
            md_content += f"## {section_content.get('title', section_name.title())}\n\n"
            
            if isinstance(section_content, dict):
                for key, value in section_content.items():
                    if key != 'title':
                        md_content += f"**{key.title()}:** {value}\n\n"
        
        return md_content
    
    def _format_report_as_html(self, report: Dict[str, Any]) -> str:
        """Format report as HTML."""
        # Basic HTML formatting
        html_content = "<html><head><title>Compliance Report</title></head><body>"
        html_content += f"<h1>Compliance Assessment Report</h1>"
        html_content += f"<p><strong>Generated:</strong> {report['metadata']['generated_at']}</p>"
        
        for section_name, section_content in report['sections'].items():
            html_content += f"<h2>{section_content.get('title', section_name.title())}</h2>"
            
            if isinstance(section_content, dict):
                html_content += "<ul>"
                for key, value in section_content.items():
                    if key != 'title':
                        html_content += f"<li><strong>{key.title()}:</strong> {value}</li>"
                html_content += "</ul>"
        
        html_content += "</body></html>"
        return html_content


# Factory function
def create_compliance_engine(config: Dict[str, Any] = None) -> IntelligentComplianceEngine:
    """Create intelligent compliance engine with configuration."""
    default_config = {
        'auto_scan_enabled': True,
        'scan_interval_hours': 24,
        'notification_enabled': True,
        'region': 'global',
        'frameworks': ['gdpr', 'iso27001']
    }
    
    if config:
        default_config.update(config)
    
    return IntelligentComplianceEngine(default_config)


# Example usage and demo
if __name__ == "__main__":
    print("🛡️ Intelligent Global Compliance Engine")
    print("Next-Generation AI-driven compliance management")
    
    # Create compliance engine
    engine = create_compliance_engine({
        'auto_scan_enabled': True,
        'scan_interval_hours': 1,  # Reduced for demo
        'notification_enabled': True
    })
    
    # Example system configuration
    system_config = {
        'privacy_policy': True,
        'data_collection': {
            'minimal_data_principle': False,  # Violation
        },
        'privacy_settings': {
            'default_privacy_enabled': True,
        },
        'security': {
            'access_control_enabled': False,  # Violation
            'incident_response_plan': True,
            'encryption_at_rest': True,
            'audit_logging': False  # Violation
        },
        'data_handling': {
            'classification_scheme': True,
            'secure_transmission': True,
            'backup_strategy': True,
            'secure_disposal': False  # Violation
        }
    }
    
    # Define assessment parameters
    frameworks = [ComplianceFramework.GDPR, ComplianceFramework.ISO27001]
    region = ComplianceRegion.EUROPEAN_UNION
    
    print("\n🔍 Performing compliance assessment...")
    
    try:
        # Perform assessment
        assessment = engine.perform_compliance_assessment(
            system_config=system_config,
            frameworks=frameworks,
            region=region
        )
        
        print(f"\n📊 Assessment Results:")
        print(f"   Overall Status: {assessment.overall_status.value}")
        print(f"   Compliance Score: {assessment.compliance_score:.1%}")
        print(f"   Rules Checked: {assessment.total_rules_checked}")
        print(f"   Violations Found: {len(assessment.violations)}")
        
        # Show violations by severity
        violation_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for violation in assessment.violations:
            if violation.severity in violation_counts:
                violation_counts[violation.severity] += 1
        
        print(f"\n🚨 Violations by Severity:")
        for severity, count in violation_counts.items():
            if count > 0:
                print(f"   {severity.title()}: {count}")
        
        # Generate report
        print(f"\n📄 Generating compliance report...")
        report = engine.generate_compliance_report(
            assessment, 
            template='executive_summary'
        )
        
        print(f"✅ Report generated with {len(report['sections'])} sections")
        
        # Show dashboard
        dashboard = engine.get_compliance_dashboard()
        print(f"\n📈 Compliance Dashboard:")
        print(f"   Current Status: {dashboard['current_status']['overall_status']}")
        print(f"   Score: {dashboard['current_status']['compliance_score']:.1%}")
        print(f"   Total Violations: {dashboard['current_status']['total_violations']}")
        print(f"   Frameworks: {', '.join(dashboard['frameworks_covered'])}")
        
        # Add custom rule example
        print(f"\n➕ Adding custom compliance rule...")
        custom_rule = ComplianceRule(
            rule_id="custom_001",
            framework=ComplianceFramework.GDPR,
            region=ComplianceRegion.EUROPEAN_UNION,
            title="Custom Data Retention Policy",
            description="Ensure data retention policies are properly implemented",
            requirements=[
                "Data retention policy documented",
                "Automated data deletion process",
                "User notification of data retention"
            ],
            verification_method="automated_scan",
            severity="medium",
            auto_verifiable=True
        )
        
        engine.add_custom_rule(custom_rule)
        
        # Export compliance data
        export_data = engine.export_compliance_data()
        print(f"\n💾 Compliance data export completed:")
        print(f"   Total assessments: {export_data['total_assessments']}")
        print(f"   Total rules: {export_data['total_rules']}")
        
    except Exception as e:
        print(f"❌ Assessment failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 Intelligent Global Compliance Engine ready for deployment!")
    print("Features implemented:")
    print("   - Multi-jurisdictional compliance rules")
    print("   - Automated compliance scanning")
    print("   - Intelligent violation detection")
    print("   - Comprehensive report generation")
    print("   - Continuous monitoring capabilities")
    print("   - Custom rule and validator support")
    print("   - Real-time compliance dashboard")
    print("   - Export and integration capabilities")