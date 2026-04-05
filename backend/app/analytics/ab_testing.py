"""
VisionX Analytics Layer - A/B Testing Framework

Experimentation framework for testing new features, models, and UX changes.
Supports variant assignment, metric tracking, and statistical significance testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
import hashlib
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment lifecycle status"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ABTestFramework:
    """
    A/B Testing framework for VisionX experiments
    
    Features:
    - Create multi-variant experiments (A/B/C/D...)
    - Deterministic user assignment (consistent variants)
    - Metric tracking per variant
    - Statistical significance testing (t-test, chi-square)
    - Sample size recommendations
    - Early stopping criteria
    """
    
    def __init__(self):
        """Initialize A/B testing framework"""
        self.experiments = {}
        self.assignments = {}  # user_id → {experiment_id → variant}
        self.metrics = pd.DataFrame()
        logger.info("✅ A/B Testing Framework initialized")
    
    
    def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        metric_name: str,
        metric_type: str = "continuous",  # or "binary"
        traffic_allocation: float = 1.0,  # % of users in experiment
        start_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Create new A/B test experiment
        
        Args:
            name: Experiment name (unique identifier)
            description: What's being tested
            variants: List of variants, e.g.:
                [
                    {'name': 'control', 'description': 'Current version', 'weight': 0.5},
                    {'name': 'treatment', 'description': 'New version', 'weight': 0.5}
                ]
            metric_name: Primary metric to optimize (e.g., 'conversion_rate')
            metric_type: 'continuous' (avg) or 'binary' (rate)
            traffic_allocation: % of users exposed (0.0 to 1.0)
            start_date: When experiment starts (default: now)
            
        Returns:
            {
                'experiment_id': str,
                'name': str,
                'status': str,
                'variants': [...],
                'created_at': str
            }
        """
        if name in self.experiments:
            raise ValueError(f"Experiment '{name}' already exists")
        
        # Validate variant weights sum to 1.0
        total_weight = sum(v.get('weight', 1.0 / len(variants)) for v in variants)
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Variant weights must sum to 1.0 (got {total_weight})")
        
        # Create experiment
        experiment = {
            'experiment_id': name,
            'name': name,
            'description': description,
            'variants': variants,
            'metric_name': metric_name,
            'metric_type': metric_type,
            'traffic_allocation': traffic_allocation,
            'status': ExperimentStatus.DRAFT,
            'start_date': start_date or datetime.now(),
            'end_date': None,
            'created_at': datetime.now(),
            'sample_size_per_variant': {},
            'results': None
        }
        
        self.experiments[name] = experiment
        logger.info(f"Created experiment: {name} with {len(variants)} variants")
        
        return {
            'experiment_id': experiment['experiment_id'],
            'name': experiment['name'],
            'status': experiment['status'].value,
            'variants': experiment['variants'],
            'created_at': experiment['created_at'].isoformat()
        }
    
    
    def start_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Start running an experiment
        
        Args:
            experiment_id: Experiment to start
            
        Returns:
            Updated experiment info
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment '{experiment_id}' not found")
        
        experiment = self.experiments[experiment_id]
        experiment['status'] = ExperimentStatus.RUNNING
        experiment['start_date'] = datetime.now()
        
        logger.info(f"Started experiment: {experiment_id}")
        
        return self.get_experiment_info(experiment_id)
    
    
    def assign_variant(self, experiment_id: str, user_id: str) -> str:
        """
        Assign user to experiment variant (deterministic)
        
        Uses consistent hashing to ensure same user always gets same variant
        
        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            
        Returns:
            Variant name assigned to user
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment '{experiment_id}' not found")
        
        experiment = self.experiments[experiment_id]
        
        # Check if user already assigned
        if user_id in self.assignments:
            if experiment_id in self.assignments[user_id]:
                return self.assignments[user_id][experiment_id]
        
        # Check if user should be included (traffic allocation)
        user_hash = int(hashlib.md5(f"{user_id}_{experiment_id}".encode()).hexdigest(), 16)
        inclusion_threshold = experiment['traffic_allocation']
        
        if (user_hash % 10000) / 10000.0 > inclusion_threshold:
            # User not in experiment
            return "control"  # Default to control if not included
        
        # Assign variant based on weights
        variants = experiment['variants']
        weights = [v.get('weight', 1.0 / len(variants)) for v in variants]
        cumulative_weights = np.cumsum(weights)
        
        # Use consistent hashing for variant assignment
        variant_hash = (user_hash % 10000) / 10000.0
        variant_index = np.searchsorted(cumulative_weights, variant_hash)
        assigned_variant = variants[variant_index]['name']
        
        # Store assignment
        if user_id not in self.assignments:
            self.assignments[user_id] = {}
        self.assignments[user_id][experiment_id] = assigned_variant
        
        logger.debug(f"Assigned user {user_id} to variant '{assigned_variant}' in experiment '{experiment_id}'")
        
        return assigned_variant
    
    
    def track_metric(
        self,
        experiment_id: str,
        user_id: str,
        metric_value: float,
        timestamp: Optional[datetime] = None
    ):
        """
        Track metric for user in experiment
        
        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            metric_value: Metric value (e.g., 1.0 for conversion, 180.5 for decision time)
            timestamp: When metric was recorded (default: now)
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment '{experiment_id}' not found")
        
        # Get variant assignment
        variant = self.assign_variant(experiment_id, user_id)
        
        # Store metric
        metric_entry = pd.DataFrame([{
            'timestamp': timestamp or datetime.now(),
            'experiment_id': experiment_id,
            'user_id': user_id,
            'variant': variant,
            'metric_value': metric_value
        }])
        
        self.metrics = pd.concat([self.metrics, metric_entry], ignore_index=True)
        
        logger.debug(f"Tracked metric for user {user_id} in {experiment_id}: {metric_value}")
    
    
    def analyze_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Analyze experiment results with statistical significance testing
        
        Args:
            experiment_id: Experiment to analyze
            
        Returns:
            {
                'experiment_id': str,
                'variants': {
                    'control': {
                        'sample_size': int,
                        'mean': float,
                        'std': float,
                        'conversion_rate': float  # if binary
                    },
                    'treatment': {...}
                },
                'comparisons': [
                    {
                        'variant_a': 'control',
                        'variant_b': 'treatment',
                        'p_value': float,
                        'significant': bool,
                        'confidence_level': float,
                        'lift': float,  # % improvement
                        'recommendation': str
                    }
                ],
                'winner': str or None,
                'status': str
            }
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment '{experiment_id}' not found")
        
        experiment = self.experiments[experiment_id]
        
        # Filter metrics for this experiment
        exp_metrics = self.metrics[self.metrics['experiment_id'] == experiment_id].copy()
        
        if exp_metrics.empty:
            return {
                'experiment_id': experiment_id,
                'error': 'No data collected yet',
                'status': experiment['status'].value
            }
        
        # Compute stats per variant
        variant_stats = {}
        for variant_name in exp_metrics['variant'].unique():
            variant_data = exp_metrics[exp_metrics['variant'] == variant_name]['metric_value']
            
            variant_stats[variant_name] = {
                'sample_size': len(variant_data),
                'mean': float(variant_data.mean()),
                'std': float(variant_data.std()),
                'median': float(variant_data.median()),
                'min': float(variant_data.min()),
                'max': float(variant_data.max())
            }
            
            # For binary metrics, compute conversion rate
            if experiment['metric_type'] == 'binary':
                variant_stats[variant_name]['conversion_rate'] = float(variant_data.mean())
        
        # Compare variants (pairwise)
        comparisons = []
        variant_names = list(variant_stats.keys())
        
        # Use first variant as control (typically 'control' or 'A')
        control_variant = variant_names[0]
        control_data = exp_metrics[exp_metrics['variant'] == control_variant]['metric_value']
        
        for treatment_variant in variant_names[1:]:
            treatment_data = exp_metrics[exp_metrics['variant'] == treatment_variant]['metric_value']
            
            # Statistical test
            if experiment['metric_type'] == 'continuous':
                # Independent t-test
                t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
            else:
                # Chi-square test for binary
                control_success = int(control_data.sum())
                control_total = len(control_data)
                treatment_success = int(treatment_data.sum())
                treatment_total = len(treatment_data)
                
                contingency = np.array([
                    [control_success, control_total - control_success],
                    [treatment_success, treatment_total - treatment_success]
                ])
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            # Compute lift
            control_mean = variant_stats[control_variant]['mean']
            treatment_mean = variant_stats[treatment_variant]['mean']
            lift = ((treatment_mean - control_mean) / control_mean) * 100 if control_mean != 0 else 0
            
            # Determine significance (p < 0.05)
            significant = p_value < 0.05
            confidence_level = (1 - p_value) * 100
            
            # Generate recommendation
            if significant:
                if lift > 0:
                    recommendation = f"✅ {treatment_variant} significantly better ({lift:.1f}% lift)"
                else:
                    recommendation = f"❌ {treatment_variant} significantly worse ({lift:.1f}% drop)"
            else:
                recommendation = f"⚠️ No significant difference (p={p_value:.3f})"
            
            comparisons.append({
                'variant_a': control_variant,
                'variant_b': treatment_variant,
                'p_value': float(p_value),
                'significant': significant,
                'confidence_level': float(confidence_level),
                'lift': float(lift),
                'recommendation': recommendation
            })
        
        # Determine winner
        winner = None
        significant_improvements = [
            c for c in comparisons 
            if c['significant'] and c['lift'] > 0
        ]
        
        if significant_improvements:
            best = max(significant_improvements, key=lambda x: x['lift'])
            winner = best['variant_b']
        
        return {
            'experiment_id': experiment_id,
            'status': experiment['status'].value,
            'metric_name': experiment['metric_name'],
            'metric_type': experiment['metric_type'],
            'variants': variant_stats,
            'comparisons': comparisons,
            'winner': winner,
            'analyzed_at': datetime.now().isoformat()
        }
    
    
    def get_experiment_info(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment metadata and current status"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment '{experiment_id}' not found")
        
        experiment = self.experiments[experiment_id]
        
        # Count assignments per variant
        exp_assignments = {}
        for user_id, user_exp in self.assignments.items():
            if experiment_id in user_exp:
                variant = user_exp[experiment_id]
                exp_assignments[variant] = exp_assignments.get(variant, 0) + 1
        
        return {
            'experiment_id': experiment['experiment_id'],
            'name': experiment['name'],
            'description': experiment['description'],
            'status': experiment['status'].value,
            'variants': experiment['variants'],
            'metric_name': experiment['metric_name'],
            'traffic_allocation': experiment['traffic_allocation'],
            'start_date': experiment['start_date'].isoformat() if experiment['start_date'] else None,
            'users_enrolled': sum(exp_assignments.values()),
            'variant_distribution': exp_assignments,
            'created_at': experiment['created_at'].isoformat()
        }
    
    
    def list_experiments(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all experiments, optionally filtered by status
        
        Args:
            status: Filter by status ('running', 'completed', etc.)
            
        Returns:
            List of experiment info dicts
        """
        experiments = list(self.experiments.values())
        
        if status:
            status_enum = ExperimentStatus(status.lower())
            experiments = [e for e in experiments if e['status'] == status_enum]
        
        return [
            self.get_experiment_info(e['experiment_id'])
            for e in experiments
        ]
    
    
    def compute_required_sample_size(
        self,
        baseline_conversion: float,
        mde: float,  # Minimum Detectable Effect (e.g., 0.05 for 5%)
        alpha: float = 0.05,  # Significance level
        power: float = 0.80  # Statistical power
    ) -> int:
        """
        Compute required sample size per variant
        
        Args:
            baseline_conversion: Current conversion rate (e.g., 0.12 for 12%)
            mde: Minimum effect you want to detect (e.g., 0.05 for 5% relative lift)
            alpha: Significance level (default: 0.05)
            power: Statistical power (default: 0.80)
            
        Returns:
            Required sample size per variant
        """
        from scipy.stats import norm
        
        # Expected conversion rates
        p1 = baseline_conversion
        p2 = p1 * (1 + mde)
        
        # Pooled proportion
        p_pooled = (p1 + p2) / 2
        
        # Z-scores
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        
        # Sample size formula
        n = (
            (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) +
             z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        ) / ((p2 - p1) ** 2)
        
        return int(np.ceil(n))


# Example usage
if __name__ == "__main__":
    ab_test = ABTestFramework()
    
    # Create experiment
    experiment = ab_test.create_experiment(
        name="shap_explainability_test",
        description="Test impact of SHAP explanations on user confidence",
        variants=[
            {'name': 'control', 'description': 'No explanations', 'weight': 0.5},
            {'name': 'treatment', 'description': 'Show SHAP explanations', 'weight': 0.5}
        ],
        metric_name='user_confidence',
        metric_type='continuous'
    )
    
    print(f"Created experiment: {experiment['name']}")
    
    # Start experiment
    ab_test.start_experiment("shap_explainability_test")
    
    # Simulate user data
    for i in range(100):
        user_id = f"user_{i}"
        variant = ab_test.assign_variant("shap_explainability_test", user_id)
        
        # Simulate metric (treatment has +10% lift)
        base_confidence = np.random.normal(0.75, 0.15)
        if variant == 'treatment':
            metric_value = base_confidence * 1.10
        else:
            metric_value = base_confidence
        
        ab_test.track_metric("shap_explainability_test", user_id, metric_value)
    
    # Analyze results
    results = ab_test.analyze_results("shap_explainability_test")
    
    print("\n📊 A/B Test Results:")
    print(f"Winner: {results.get('winner', 'No clear winner')}")
    for comparison in results['comparisons']:
        print(f"\n{comparison['recommendation']}")
        print(f"  P-value: {comparison['p_value']:.4f}")
        print(f"  Lift: {comparison['lift']:.1f}%")
