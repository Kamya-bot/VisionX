"""
VisionX Analytics Layer - KPI Tracking System

Real-time KPI computation, storage, and trend analysis for business intelligence.
Tracks key performance indicators across the ML platform.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class KPIStatus(Enum):
    """KPI health status"""
    EXCELLENT = "excellent"  # > target by 20%+
    GOOD = "good"  # > target
    WARNING = "warning"  # < target but > critical
    CRITICAL = "critical"  # < critical threshold


class KPITracker:
    """
    Real-time KPI tracking and trend analysis
    
    Features:
    - Track 15+ key metrics
    - Historical storage (30-day window)
    - Trend detection (improving/declining/stable)
    - Alert thresholds (target vs. critical)
    - Executive dashboard data
    """
    
    def __init__(self):
        """Initialize KPI tracker with empty state"""
        self.kpis = {}
        self.history = pd.DataFrame()
        self.targets = self._initialize_targets()
        logger.info("✅ KPI Tracker initialized")
    
    
    def _initialize_targets(self) -> Dict[str, Dict[str, float]]:
        """
        Define KPI targets and critical thresholds
        
        Returns:
            {
                'kpi_name': {
                    'target': float,  # Target value
                    'critical': float,  # Critical threshold
                    'direction': 'higher' or 'lower'  # Good direction
                }
            }
        """
        return {
            # User Engagement KPIs
            'avg_session_time': {
                'target': 300,  # 5 minutes
                'critical': 120,  # 2 minutes
                'direction': 'higher'
            },
            'avg_clicks_per_session': {
                'target': 30,
                'critical': 10,
                'direction': 'higher'
            },
            'avg_scroll_depth': {
                'target': 0.75,  # 75% page scroll
                'critical': 0.40,
                'direction': 'higher'
            },
            
            # ML Performance KPIs
            'prediction_accuracy': {
                'target': 0.87,  # 87%
                'critical': 0.75,  # 75%
                'direction': 'higher'
            },
            'avg_confidence': {
                'target': 0.80,  # 80%
                'critical': 0.60,  # 60%
                'direction': 'higher'
            },
            'high_confidence_rate': {
                'target': 0.70,  # 70% of predictions > 0.8
                'critical': 0.50,
                'direction': 'higher'
            },
            
            # Decision Quality KPIs
            'avg_decision_time': {
                'target': 180,  # 3 minutes
                'critical': 300,  # 5 minutes
                'direction': 'lower'  # Lower is better
            },
            'decision_efficiency': {
                'target': 0.05,  # product_views / decision_time
                'critical': 0.02,
                'direction': 'higher'
            },
            
            # Business KPIs
            'conversion_rate': {
                'target': 0.12,  # 12%
                'critical': 0.08,  # 8%
                'direction': 'higher'
            },
            'user_satisfaction': {
                'target': 0.80,  # 80%
                'critical': 0.60,
                'direction': 'higher'
            },
            
            # Technical KPIs
            'api_latency_p95': {
                'target': 150,  # 150ms
                'critical': 300,  # 300ms
                'direction': 'lower'
            },
            'api_error_rate': {
                'target': 0.01,  # 1%
                'critical': 0.05,  # 5%
                'direction': 'lower'
            },
            
            # Feature Usage KPIs
            'explainability_usage_rate': {
                'target': 0.40,  # 40% of users
                'critical': 0.20,
                'direction': 'higher'
            },
            'simulation_usage_rate': {
                'target': 0.30,  # 30% of users
                'critical': 0.15,
                'direction': 'higher'
            },
            
            # Cluster Performance KPIs
            'power_users_percentage': {
                'target': 0.25,  # 25% Power Decision Makers
                'critical': 0.10,
                'direction': 'higher'
            }
        }
    
    
    def update_kpi(self, kpi_name: str, value: float, timestamp: Optional[datetime] = None):
        """
        Update KPI value and store in history
        
        Args:
            kpi_name: Name of KPI
            value: Current value
            timestamp: When measured (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store current value
        self.kpis[kpi_name] = {
            'value': value,
            'timestamp': timestamp,
            'status': self._compute_status(kpi_name, value)
        }
        
        # Add to history
        history_entry = pd.DataFrame([{
            'timestamp': timestamp,
            'kpi_name': kpi_name,
            'value': value
        }])
        self.history = pd.concat([self.history, history_entry], ignore_index=True)
        
        logger.debug(f"Updated KPI: {kpi_name} = {value}")
    
    
    def _compute_status(self, kpi_name: str, value: float) -> KPIStatus:
        """
        Compute KPI health status
        
        Args:
            kpi_name: Name of KPI
            value: Current value
            
        Returns:
            KPIStatus (EXCELLENT/GOOD/WARNING/CRITICAL)
        """
        if kpi_name not in self.targets:
            return KPIStatus.GOOD  # Default
        
        target_config = self.targets[kpi_name]
        target = target_config['target']
        critical = target_config['critical']
        direction = target_config['direction']
        
        if direction == 'higher':
            # Higher is better
            if value >= target * 1.2:
                return KPIStatus.EXCELLENT
            elif value >= target:
                return KPIStatus.GOOD
            elif value >= critical:
                return KPIStatus.WARNING
            else:
                return KPIStatus.CRITICAL
        else:
            # Lower is better
            if value <= target * 0.8:
                return KPIStatus.EXCELLENT
            elif value <= target:
                return KPIStatus.GOOD
            elif value <= critical:
                return KPIStatus.WARNING
            else:
                return KPIStatus.CRITICAL
    
    
    def get_kpi(self, kpi_name: str) -> Dict[str, Any]:
        """
        Get current KPI value with status
        
        Returns:
            {
                'name': str,
                'value': float,
                'status': str,
                'target': float,
                'critical': float,
                'timestamp': str
            }
        """
        if kpi_name not in self.kpis:
            return {"error": f"KPI {kpi_name} not found"}
        
        kpi_data = self.kpis[kpi_name]
        target_config = self.targets.get(kpi_name, {})
        
        return {
            'name': kpi_name,
            'value': kpi_data['value'],
            'status': kpi_data['status'].value,
            'target': target_config.get('target'),
            'critical': target_config.get('critical'),
            'timestamp': kpi_data['timestamp'].isoformat()
        }
    
    
    def get_all_kpis(self) -> Dict[str, Any]:
        """
        Get all current KPIs grouped by category
        
        Returns:
            {
                'user_engagement': [...],
                'ml_performance': [...],
                'decision_quality': [...],
                'business': [...],
                'technical': [...],
                'feature_usage': [...],
                'summary': {...}
            }
        """
        # Define categories
        categories = {
            'user_engagement': [
                'avg_session_time', 'avg_clicks_per_session', 'avg_scroll_depth'
            ],
            'ml_performance': [
                'prediction_accuracy', 'avg_confidence', 'high_confidence_rate'
            ],
            'decision_quality': [
                'avg_decision_time', 'decision_efficiency'
            ],
            'business': [
                'conversion_rate', 'user_satisfaction'
            ],
            'technical': [
                'api_latency_p95', 'api_error_rate'
            ],
            'feature_usage': [
                'explainability_usage_rate', 'simulation_usage_rate'
            ]
        }
        
        # Group KPIs by category
        result = {}
        for category, kpi_list in categories.items():
            result[category] = [
                self.get_kpi(kpi_name)
                for kpi_name in kpi_list
                if kpi_name in self.kpis
            ]
        
        # Add summary
        result['summary'] = self._compute_summary()
        
        return result
    
    
    def _compute_summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics across all KPIs
        
        Returns:
            {
                'total_kpis': int,
                'excellent_count': int,
                'good_count': int,
                'warning_count': int,
                'critical_count': int,
                'health_score': float  # 0-100
            }
        """
        if not self.kpis:
            return {
                'total_kpis': 0,
                'excellent_count': 0,
                'good_count': 0,
                'warning_count': 0,
                'critical_count': 0,
                'health_score': 0.0
            }
        
        statuses = [kpi['status'] for kpi in self.kpis.values()]
        
        counts = {
            'excellent_count': sum(1 for s in statuses if s == KPIStatus.EXCELLENT),
            'good_count': sum(1 for s in statuses if s == KPIStatus.GOOD),
            'warning_count': sum(1 for s in statuses if s == KPIStatus.WARNING),
            'critical_count': sum(1 for s in statuses if s == KPIStatus.CRITICAL)
        }
        
        # Compute health score (weighted average)
        total = len(statuses)
        health_score = (
            (counts['excellent_count'] * 100 +
             counts['good_count'] * 75 +
             counts['warning_count'] * 50 +
             counts['critical_count'] * 25) / total
        ) if total > 0 else 0.0
        
        return {
            'total_kpis': total,
            **counts,
            'health_score': round(health_score, 2)
        }
    
    
    def get_kpi_trend(self, kpi_name: str, days: int = 7) -> Dict[str, Any]:
        """
        Analyze KPI trend over time
        
        Args:
            kpi_name: Name of KPI
            days: Number of days to analyze
            
        Returns:
            {
                'kpi_name': str,
                'trend': 'improving'|'declining'|'stable',
                'change_percent': float,
                'current_value': float,
                'previous_value': float,
                'data_points': int,
                'time_series': [{timestamp, value}, ...]
            }
        """
        if self.history.empty:
            return {"error": "No historical data available"}
        
        # Filter history for this KPI
        kpi_history = self.history[self.history['kpi_name'] == kpi_name].copy()
        
        if kpi_history.empty:
            return {"error": f"No history for KPI: {kpi_name}"}
        
        # Filter by time range
        cutoff_date = datetime.now() - timedelta(days=days)
        kpi_history = kpi_history[kpi_history['timestamp'] >= cutoff_date]
        
        if len(kpi_history) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Sort by timestamp
        kpi_history = kpi_history.sort_values('timestamp')
        
        # Get current and previous values
        current_value = kpi_history.iloc[-1]['value']
        previous_value = kpi_history.iloc[0]['value']
        
        # Compute change
        change_percent = ((current_value - previous_value) / previous_value) * 100 if previous_value != 0 else 0
        
        # Determine trend direction
        target_config = self.targets.get(kpi_name, {})
        direction = target_config.get('direction', 'higher')
        
        if abs(change_percent) < 5:
            trend = 'stable'
        elif (direction == 'higher' and change_percent > 0) or (direction == 'lower' and change_percent < 0):
            trend = 'improving'
        else:
            trend = 'declining'
        
        # Prepare time series data
        time_series = kpi_history[['timestamp', 'value']].to_dict('records')
        
        return {
            'kpi_name': kpi_name,
            'trend': trend,
            'change_percent': round(change_percent, 2),
            'current_value': round(current_value, 3),
            'previous_value': round(previous_value, 3),
            'data_points': len(kpi_history),
            'time_series': time_series
        }
    
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Get list of KPIs in WARNING or CRITICAL status
        
        Returns:
            [
                {
                    'kpi_name': str,
                    'status': str,
                    'value': float,
                    'target': float,
                    'message': str
                },
                ...
            ]
        """
        alerts = []
        
        for kpi_name, kpi_data in self.kpis.items():
            status = kpi_data['status']
            
            if status in [KPIStatus.WARNING, KPIStatus.CRITICAL]:
                target_config = self.targets.get(kpi_name, {})
                
                message = f"{kpi_name} is {status.value}: {kpi_data['value']:.3f}"
                if 'target' in target_config:
                    message += f" (target: {target_config['target']})"
                
                alerts.append({
                    'kpi_name': kpi_name,
                    'status': status.value,
                    'value': kpi_data['value'],
                    'target': target_config.get('target'),
                    'critical': target_config.get('critical'),
                    'message': message
                })
        
        return alerts
    
    
    def generate_kpi_report(self, days: int = 7) -> Dict[str, Any]:
        """
        Generate comprehensive KPI report
        
        Returns:
            {
                'generated_at': str,
                'time_range_days': int,
                'summary': {...},
                'kpis_by_category': {...},
                'trends': [...],
                'alerts': [...],
                'recommendations': [...]
            }
        """
        all_kpis = self.get_all_kpis()
        
        # Get trends for all KPIs
        trends = []
        for kpi_name in self.kpis.keys():
            trend = self.get_kpi_trend(kpi_name, days=days)
            if 'error' not in trend:
                trends.append(trend)
        
        # Get alerts
        alerts = self.get_alerts()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(alerts, trends)
        
        return {
            'generated_at': datetime.now().isoformat(),
            'time_range_days': days,
            'summary': all_kpis['summary'],
            'kpis_by_category': {
                k: v for k, v in all_kpis.items() if k != 'summary'
            },
            'trends': trends,
            'alerts': alerts,
            'recommendations': recommendations
        }
    
    
    def _generate_recommendations(self, alerts: List[Dict], trends: List[Dict]) -> List[str]:
        """
        Generate actionable recommendations based on KPIs
        
        Args:
            alerts: List of KPI alerts
            trends: List of KPI trends
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check for critical alerts
        critical_alerts = [a for a in alerts if a['status'] == 'critical']
        if critical_alerts:
            for alert in critical_alerts:
                recommendations.append(
                    f"URGENT: Address {alert['kpi_name']} - currently {alert['value']:.3f}, "
                    f"critical threshold is {alert['critical']}"
                )
        
        # Check for declining trends
        declining = [t for t in trends if t['trend'] == 'declining']
        if declining:
            for trend in declining[:3]:  # Top 3
                recommendations.append(
                    f"Monitor {trend['kpi_name']} - declining by {abs(trend['change_percent']):.1f}% "
                    f"over past 7 days"
                )
        
        # Check for improving trends (positive feedback)
        improving = [t for t in trends if t['trend'] == 'improving']
        if improving:
            best_improvement = max(improving, key=lambda x: abs(x['change_percent']))
            recommendations.append(
                f"Continue efforts on {best_improvement['kpi_name']} - "
                f"improving by {abs(best_improvement['change_percent']):.1f}%"
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append("All KPIs are healthy. Maintain current strategies.")
        
        return recommendations


# Example usage
if __name__ == "__main__":
    tracker = KPITracker()
    
    # Simulate some KPI updates
    tracker.update_kpi('prediction_accuracy', 0.873)
    tracker.update_kpi('avg_confidence', 0.82)
    tracker.update_kpi('avg_decision_time', 175)
    tracker.update_kpi('conversion_rate', 0.125)
    tracker.update_kpi('api_latency_p95', 140)
    
    # Get report
    report = tracker.generate_kpi_report()
    
    print("📊 KPI Report:")
    print(f"Health Score: {report['summary']['health_score']}/100")
    print(f"Alerts: {len(report['alerts'])}")
    print(f"Recommendations: {len(report['recommendations'])}")
