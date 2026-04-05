"""
VisionX - Real-Time Data & Prediction Drift Detection
Production-grade ML monitoring for model reliability
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects data drift and prediction drift using statistical tests
    
    Features:
    - Kolmogorov-Smirnov test for feature distribution changes
    - Population Stability Index (PSI) for prediction drift
    - Configurable thresholds and time windows
    - Real-time alerting capabilities
    """
    
    def __init__(
        self,
        ks_threshold: float = 0.05,  # p-value threshold for KS test
        psi_threshold: float = 0.1,   # PSI threshold (0.1 = minor shift)
        lookback_days: int = 7
    ):
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.lookback_days = lookback_days
        
    def detect_feature_drift(
        self,
        baseline_features: np.ndarray,
        current_features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Dict]:
        """
        Detect drift in individual features using Kolmogorov-Smirnov test
        
        Args:
            baseline_features: Historical feature values (N x M)
            current_features: Recent feature values (N x M)
            feature_names: List of feature names
            
        Returns:
            Dict with drift results per feature
        """
        drift_results = {}
        
        for i, feature_name in enumerate(feature_names):
            baseline = baseline_features[:, i]
            current = current_features[:, i]
            
            # KS test
            ks_stat, p_value = stats.ks_2samp(baseline, current)
            
            # Drift detected if p-value < threshold
            is_drift = p_value < self.ks_threshold
            
            # Calculate magnitude of drift
            mean_shift = abs(np.mean(current) - np.mean(baseline))
            std_shift = abs(np.std(current) - np.std(baseline))
            
            drift_results[feature_name] = {
                'ks_statistic': float(ks_stat),
                'p_value': float(p_value),
                'is_drift': is_drift,
                'mean_shift': float(mean_shift),
                'std_shift': float(std_shift),
                'severity': self._calculate_severity(ks_stat, p_value),
                'baseline_mean': float(np.mean(baseline)),
                'current_mean': float(np.mean(current)),
                'baseline_std': float(np.std(baseline)),
                'current_std': float(np.std(current))
            }
            
        return drift_results
    
    def detect_prediction_drift(
        self,
        baseline_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> Dict:
        """
        Detect drift in model predictions using Population Stability Index (PSI)
        
        Args:
            baseline_predictions: Historical predictions
            current_predictions: Recent predictions
            
        Returns:
            Dict with PSI score and drift status
        """
        # Calculate PSI
        psi = self._calculate_psi(baseline_predictions, current_predictions)
        
        # Determine drift severity
        if psi < 0.1:
            severity = 'none'
            status = 'stable'
        elif psi < 0.25:
            severity = 'minor'
            status = 'monitor'
        else:
            severity = 'major'
            status = 'alert'
        
        return {
            'psi_score': float(psi),
            'is_drift': psi > self.psi_threshold,
            'severity': severity,
            'status': status,
            'baseline_mean': float(np.mean(baseline_predictions)),
            'current_mean': float(np.mean(current_predictions)),
            'baseline_std': float(np.std(baseline_predictions)),
            'current_std': float(np.std(current_predictions))
        }
    
    def _calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        PSI measures how much a distribution has shifted
        PSI < 0.1: No significant change
        PSI 0.1-0.25: Minor shift
        PSI > 0.25: Major shift (model may need retraining)
        """
        # Create bins based on baseline distribution
        breakpoints = np.linspace(
            min(baseline.min(), current.min()),
            max(baseline.max(), current.max()),
            bins + 1
        )
        
        # Calculate distributions
        baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
        current_counts = np.histogram(current, bins=breakpoints)[0]
        
        # Convert to proportions (add small epsilon to avoid division by zero)
        epsilon = 1e-10
        baseline_props = baseline_counts / (len(baseline) + epsilon)
        current_props = current_counts / (len(current) + epsilon)
        
        # Calculate PSI
        psi = np.sum(
            (current_props - baseline_props) * 
            np.log((current_props + epsilon) / (baseline_props + epsilon))
        )
        
        return abs(psi)
    
    def _calculate_severity(self, ks_stat: float, p_value: float) -> str:
        """Calculate drift severity level"""
        if p_value >= self.ks_threshold:
            return 'none'
        elif ks_stat < 0.1:
            return 'minor'
        elif ks_stat < 0.3:
            return 'moderate'
        else:
            return 'severe'
    
    def get_drift_summary(
        self,
        feature_drift: Dict,
        prediction_drift: Dict
    ) -> Dict:
        """
        Generate comprehensive drift summary
        
        Returns:
            Summary with overall status and recommendations
        """
        # Count drifted features
        drifted_features = [
            name for name, result in feature_drift.items()
            if result['is_drift']
        ]
        
        # Overall drift status
        has_feature_drift = len(drifted_features) > 0
        has_prediction_drift = prediction_drift['is_drift']
        
        # Determine action
        if has_prediction_drift and has_feature_drift:
            action = 'urgent_retrain'
            alert_level = 'critical'
        elif has_prediction_drift or len(drifted_features) > 2:
            action = 'investigate'
            alert_level = 'warning'
        elif len(drifted_features) > 0:
            action = 'monitor'
            alert_level = 'info'
        else:
            action = 'none'
            alert_level = 'success'
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': alert_level,
            'feature_drift_count': len(drifted_features),
            'drifted_features': drifted_features,
            'has_prediction_drift': has_prediction_drift,
            'prediction_psi': prediction_drift['psi_score'],
            'recommended_action': action,
            'summary': self._generate_summary_message(
                drifted_features,
                prediction_drift,
                action
            )
        }
    
    def _generate_summary_message(
        self,
        drifted_features: List[str],
        prediction_drift: Dict,
        action: str
    ) -> str:
        """Generate human-readable summary message"""
        if action == 'urgent_retrain':
            return (
                f"⚠️ CRITICAL: Both feature drift ({len(drifted_features)} features) "
                f"and prediction drift (PSI={prediction_drift['psi_score']:.3f}) detected. "
                "Model retraining recommended immediately."
            )
        elif action == 'investigate':
            return (
                f"⚡ WARNING: Drift detected in {len(drifted_features)} features. "
                f"Prediction PSI: {prediction_drift['psi_score']:.3f}. "
                "Investigation recommended."
            )
        elif action == 'monitor':
            return (
                f"ℹ️ INFO: Minor drift detected in {len(drifted_features)} features. "
                "Continue monitoring."
            )
        else:
            return "✅ SUCCESS: No significant drift detected. Model is stable."


class DriftMonitor:
    """
    Monitors drift over time and stores historical drift metrics
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.detector = DriftDetector()
    
    def run_drift_check(
        self,
        current_features: np.ndarray,
        current_predictions: np.ndarray,
        feature_names: List[str],
        lookback_days: int = 7
    ) -> Dict:
        """
        Run complete drift analysis comparing current data to baseline
        
        Args:
            current_features: Recent feature values
            current_predictions: Recent predictions
            feature_names: List of feature names
            lookback_days: Days to look back for baseline
            
        Returns:
            Complete drift report
        """
        try:
            # Get baseline data (would query from DB in production)
            # For now, using synthetic baseline
            baseline_features, baseline_predictions = self._get_baseline_data(
                lookback_days
            )
            
            # Detect feature drift
            feature_drift = self.detector.detect_feature_drift(
                baseline_features,
                current_features,
                feature_names
            )
            
            # Detect prediction drift
            prediction_drift = self.detector.detect_prediction_drift(
                baseline_predictions,
                current_predictions
            )
            
            # Generate summary
            summary = self.detector.get_drift_summary(
                feature_drift,
                prediction_drift
            )
            
            # Store drift metrics (would save to DB in production)
            self._store_drift_metrics(feature_drift, prediction_drift, summary)
            
            return {
                'feature_drift': feature_drift,
                'prediction_drift': prediction_drift,
                'summary': summary,
                'baseline_period': {
                    'start': (datetime.utcnow() - timedelta(days=lookback_days)).isoformat(),
                    'end': datetime.utcnow().isoformat(),
                    'days': lookback_days
                }
            }
            
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            raise
    
    def _get_baseline_data(
        self,
        lookback_days: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get baseline data from predictions_log table.
        Uses records older than lookback_days as the stable baseline window.
        Falls back to oldest 50% of all records if not enough history.
        """
        try:
            from datetime import timedelta
            import models as db_models

            cutoff = datetime.utcnow() - timedelta(days=lookback_days)
            baseline_cutoff = datetime.utcnow() - timedelta(days=lookback_days * 2)

            # Try to get older records as baseline
            rows = (
                self.db.query(db_models.PredictionLog)
                .filter(db_models.PredictionLog.created_at < cutoff)
                .filter(db_models.PredictionLog.created_at >= baseline_cutoff)
                .order_by(db_models.PredictionLog.created_at.desc())
                .limit(200)
                .all()
            )

            # If not enough history, use all records as baseline
            if len(rows) < 10:
                rows = (
                    self.db.query(db_models.PredictionLog)
                    .order_by(db_models.PredictionLog.created_at.asc())
                    .limit(100)
                    .all()
                )

            if not rows:
                # Truly no data yet — return minimal synthetic baseline with a warning
                logger.warning("No baseline data in DB; using synthetic baseline. Create more predictions.")
                n = 50
                baseline_features = np.column_stack([
                    np.random.uniform(10000, 100000, n),   # price
                    np.random.uniform(1, 10, n),           # quality_score
                    np.random.uniform(1, 10, n),           # satisfaction_score
                    np.random.uniform(0, 1, n)             # risk_score
                ])
                baseline_predictions = np.random.uniform(0.5, 0.85, n)
                return baseline_features, baseline_predictions

            feature_names = ['price', 'quality_score', 'satisfaction_score', 'risk_score']
            features_list, preds_list = [], []
            for row in rows:
                f = row.features or {}
                features_list.append([
                    float(f.get('price', 0.0)),
                    float(f.get('quality_score', 5.0)),
                    float(f.get('satisfaction_score', 5.0)),
                    float(f.get('risk_score', 0.5))
                ])
                preds_list.append(float(row.confidence))

            return np.array(features_list), np.array(preds_list)

        except Exception as e:
            logger.error(f"Failed to load baseline from DB: {e}")
            n = 50
            return np.zeros((n, 4)), np.full(n, 0.5)

    def _store_drift_metrics(
        self,
        feature_drift: Dict,
        prediction_drift: Dict,
        summary: Dict
    ):
        """
        Persist drift metrics to model_metrics table.
        """
        try:
            import models as db_models

            records = [
                db_models.ModelMetrics(
                    metric_name="drift_overall_status",
                    metric_value=1.0 if summary.get("overall_status") == "stable" else
                                 0.5 if summary.get("overall_status") == "monitor" else 0.0,
                    data_window="drift_check",
                    metadata={"status": summary.get("overall_status"), "drifted_features": summary.get("drifted_features", [])}
                ),
                db_models.ModelMetrics(
                    metric_name="drift_prediction_psi",
                    metric_value=float(summary.get("prediction_psi", 0.0)),
                    data_window="drift_check",
                    metadata={"severity": prediction_drift.get("severity", "unknown")}
                ),
                db_models.ModelMetrics(
                    metric_name="drift_feature_count",
                    metric_value=float(len(summary.get("drifted_features", []))),
                    data_window="drift_check",
                    metadata={"drifted": summary.get("drifted_features", [])}
                ),
            ]

            for rec in records:
                self.db.add(rec)
            self.db.commit()
            logger.info(f"Drift metrics persisted: status={summary.get('overall_status')}, PSI={summary.get('prediction_psi', 0):.4f}")

        except Exception as e:
            logger.error(f"Failed to store drift metrics: {e}")
            # Don't raise — drift storage failure shouldn't crash the endpoint


# Convenience function for API endpoints
def check_model_drift(
    db: Session,
    current_features: np.ndarray,
    current_predictions: np.ndarray,
    feature_names: List[str]
) -> Dict:
    """
    Run drift detection and return results
    
    Usage in API:
        drift_report = check_model_drift(db, features, predictions, feature_names)
    """
    monitor = DriftMonitor(db)
    return monitor.run_drift_check(
        current_features,
        current_predictions,
        feature_names
    )
