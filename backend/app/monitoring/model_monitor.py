"""
Model Monitoring System
Tracks model performance, detects drift, and maintains prediction logs

Why this matters:
- Catches model degradation in production
- Critical for MLOps maturity
- Companies like Amazon & Netflix require this
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from scipy import stats
import os
import logging

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Production model monitoring system
    
    Features:
    - Prediction logging
    - Confidence tracking
    - Drift detection
    - Performance metrics
    - Alert generation
    """
    
    def __init__(self, log_dir: str = "logs", alert_threshold: float = 0.6):
        """
        Initialize model monitor
        
        Args:
            log_dir: Directory to store monitoring logs
            alert_threshold: Confidence threshold for alerts
        """
        self.log_dir = log_dir
        self.alert_threshold = alert_threshold
        self.logs: List[Dict[str, Any]] = []
        self.reference_data: Optional[np.ndarray] = None
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Load existing logs if available
        self.load_logs()
        
        logger.info(f"✅ Model Monitor initialized (alert threshold: {alert_threshold})")
    
    def log_prediction(
        self,
        user_id: str,
        input_data: np.ndarray,
        prediction: Any,
        confidence: float,
        model_version: str = "1.0.0",
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Log a model prediction
        
        Args:
            user_id: User identifier
            input_data: Input feature vector
            prediction: Model prediction
            confidence: Prediction confidence score
            model_version: Version of model used
            metadata: Additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "prediction": str(prediction),
            "confidence": float(confidence),
            "model_version": model_version,
            "input_stats": {
                "mean": float(np.mean(input_data)),
                "std": float(np.std(input_data)),
                "min": float(np.min(input_data)),
                "max": float(np.max(input_data))
            }
        }
        
        if metadata:
            log_entry["metadata"] = metadata
        
        self.logs.append(log_entry)
        
        # Save logs periodically (every 10 predictions)
        if len(self.logs) % 10 == 0:
            self.save_logs()
    
    def save_logs(self) -> None:
        """Save logs to JSON file"""
        log_file = os.path.join(self.log_dir, "monitoring_logs.json")
        
        try:
            with open(log_file, "w") as f:
                json.dump(self.logs, f, indent=2)
            logger.info(f"💾 Saved {len(self.logs)} prediction logs")
        except Exception as e:
            logger.error(f"❌ Failed to save logs: {str(e)}")
    
    def load_logs(self) -> None:
        """Load existing logs from file"""
        log_file = os.path.join(self.log_dir, "monitoring_logs.json")
        
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    self.logs = json.load(f)
                logger.info(f"✅ Loaded {len(self.logs)} existing logs")
            except Exception as e:
                logger.error(f"❌ Failed to load logs: {str(e)}")
                self.logs = []
    
    def detect_drift(self, window_size: int = 100) -> Dict[str, Any]:
        """
        Detect model drift based on recent predictions
        
        Args:
            window_size: Number of recent predictions to analyze
            
        Returns:
            Drift detection report
        """
        if len(self.logs) < 10:
            return {
                "status": "insufficient_data",
                "message": "Not enough predictions for drift detection (minimum 10 required)",
                "samples_needed": 10 - len(self.logs)
            }
        
        # Get recent predictions
        recent_logs = self.logs[-window_size:]
        
        # Extract confidence scores
        confidences = [log["confidence"] for log in recent_logs]
        
        # Calculate statistics
        avg_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        min_confidence = np.min(confidences)
        
        # Detect drift
        drift_detected = False
        alerts = []
        
        # Alert 1: Low average confidence
        if avg_confidence < self.alert_threshold:
            drift_detected = True
            alerts.append({
                "type": "low_confidence",
                "severity": "high",
                "message": f"Average confidence ({avg_confidence:.2f}) below threshold ({self.alert_threshold})"
            })
        
        # Alert 2: High variance
        if std_confidence > 0.2:
            drift_detected = True
            alerts.append({
                "type": "high_variance",
                "severity": "medium",
                "message": f"High confidence variance ({std_confidence:.2f}) detected"
            })
        
        # Alert 3: Extremely low confidence
        if min_confidence < 0.4:
            drift_detected = True
            alerts.append({
                "type": "critical_confidence",
                "severity": "critical",
                "message": f"Critically low confidence ({min_confidence:.2f}) detected"
            })
        
        return {
            "drift_detected": drift_detected,
            "status": "⚠️ Drift detected" if drift_detected else "✅ Model stable",
            "metrics": {
                "avg_confidence": round(avg_confidence, 4),
                "std_confidence": round(std_confidence, 4),
                "min_confidence": round(min_confidence, 4),
                "samples_analyzed": len(recent_logs)
            },
            "alerts": alerts,
            "timestamp": datetime.now().isoformat()
        }
    
    def detect_feature_drift(
        self,
        current_data: np.ndarray,
        reference_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Detect drift in input features using statistical tests
        
        Args:
            current_data: Recent feature data (2D array)
            reference_data: Reference/training data (2D array)
            
        Returns:
            Feature drift report
        """
        if reference_data is None:
            if self.reference_data is None:
                return {
                    "status": "no_reference",
                    "message": "No reference data available for comparison"
                }
            reference_data = self.reference_data
        
        # Ensure 2D arrays
        if current_data.ndim == 1:
            current_data = current_data.reshape(-1, 1)
        if reference_data.ndim == 1:
            reference_data = reference_data.reshape(-1, 1)
        
        n_features = current_data.shape[1]
        drift_scores = []
        
        for i in range(n_features):
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(
                reference_data[:, i],
                current_data[:, i]
            )
            
            drift_detected = p_value < 0.05
            
            drift_scores.append({
                "feature_index": i,
                "ks_statistic": round(float(statistic), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": drift_detected
            })
        
        # Calculate overall drift
        n_drifted = sum(1 for score in drift_scores if score["drift_detected"])
        drift_percentage = (n_drifted / n_features) * 100
        
        return {
            "status": "drift_detected" if n_drifted > 0 else "no_drift",
            "total_features": n_features,
            "drifted_features": n_drifted,
            "drift_percentage": round(drift_percentage, 2),
            "feature_scores": drift_scores,
            "timestamp": datetime.now().isoformat()
        }
    
    def set_reference_data(self, reference_data: np.ndarray) -> None:
        """Set reference data for drift detection"""
        self.reference_data = reference_data
        logger.info(f"✅ Reference data set ({reference_data.shape[0]} samples)")
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get model performance summary for recent time window
        
        Args:
            hours: Time window in hours
            
        Returns:
            Performance summary
        """
        if not self.logs:
            return {"status": "no_data", "message": "No predictions logged"}
        
        # Filter logs by time window
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_logs = [
            log for log in self.logs
            if datetime.fromisoformat(log["timestamp"]) > cutoff_time
        ]
        
        if not recent_logs:
            return {
                "status": "no_recent_data",
                "message": f"No predictions in last {hours} hours"
            }
        
        # Calculate metrics
        confidences = [log["confidence"] for log in recent_logs]
        
        return {
            "time_window_hours": hours,
            "total_predictions": len(recent_logs),
            "avg_confidence": round(np.mean(confidences), 4),
            "median_confidence": round(np.median(confidences), 4),
            "std_confidence": round(np.std(confidences), 4),
            "min_confidence": round(np.min(confidences), 4),
            "max_confidence": round(np.max(confidences), 4),
            "predictions_per_hour": round(len(recent_logs) / hours, 2),
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report
        
        Returns:
            Complete monitoring report
        """
        drift_report = self.detect_drift()
        performance_summary = self.get_performance_summary()
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "total_predictions": len(self.logs),
            "drift_analysis": drift_report,
            "performance_24h": performance_summary,
            "status": "healthy" if not drift_report.get("drift_detected", False) else "degraded"
        }


# Example usage
if __name__ == "__main__":
    print("📊 Model Monitoring System")
    print("="*60)
    print("This system provides:")
    print("  ✅ Prediction logging")
    print("  ✅ Confidence tracking")
    print("  ✅ Drift detection")
    print("  ✅ Performance metrics")
    print("  ✅ Alert generation")
    print("\n📈 Example Alert:")
    print("  ⚠️ Model confidence dropping")
    print("  Average confidence: 0.58 (threshold: 0.60)")
    print("  Possible data drift detected")
    print("="*60)
