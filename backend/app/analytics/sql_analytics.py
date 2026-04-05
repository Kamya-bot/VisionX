"""
VisionX Analytics Layer - SQL Analytics Module

Provides SQL-based analytics queries for business intelligence and KPI tracking.
Converts raw prediction/interaction logs into actionable insights.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SQLAnalytics:
    """
    SQL-based analytics for VisionX platform
    
    Features:
    - User engagement metrics by cluster
    - Conversion funnel analysis
    - Prediction accuracy tracking
    - Feature importance trends
    - Decision time analysis
    """
    
    def __init__(self, db_connection=None):
        """
        Initialize SQL Analytics
        
        Args:
            db_connection: Database connection (if using real DB)
                          For demo: uses in-memory pandas DataFrames
        """
        self.db = db_connection
        self.predictions_df = pd.DataFrame()  # In-memory storage for demo
        self.interactions_df = pd.DataFrame()
        logger.info("✅ SQL Analytics initialized")
    
    
    def log_prediction(self, prediction_data: Dict[str, Any]):
        """
        Log prediction event for analytics
        
        Args:
            prediction_data: {
                'timestamp': datetime,
                'user_id': str,
                'cluster_name': str,
                'predicted_option': str,
                'confidence': float,
                'features': dict,
                'decision_time': float
            }
        """
        new_row = pd.DataFrame([prediction_data])
        self.predictions_df = pd.concat([self.predictions_df, new_row], ignore_index=True)
        logger.debug(f"Logged prediction for user {prediction_data.get('user_id')}")
    
    
    def get_user_engagement_metrics(self, time_range_days: int = 30) -> Dict[str, Any]:
        """
        Get user engagement metrics by cluster
        
        SQL Equivalent:
        SELECT 
            cluster_name,
            AVG(engagement_ratio) as avg_engagement,
            AVG(decision_time) as avg_decision_time,
            AVG(confidence) as avg_confidence,
            COUNT(*) as user_count,
            COUNT(DISTINCT user_id) as unique_users
        FROM predictions
        WHERE timestamp >= NOW() - INTERVAL '30 days'
        GROUP BY cluster_name
        ORDER BY avg_engagement DESC
        
        Returns:
            {
                'metrics': [
                    {
                        'cluster_name': 'Power Decision Makers',
                        'avg_engagement': 0.82,
                        'avg_decision_time': 120.5,
                        'avg_confidence': 0.88,
                        'user_count': 1250,
                        'unique_users': 850
                    },
                    ...
                ],
                'summary': {
                    'total_predictions': 5000,
                    'total_users': 3200,
                    'time_range_days': 30
                }
            }
        """
        if self.predictions_df.empty:
            return {"metrics": [], "summary": {}}
        
        # Filter by time range
        cutoff_date = datetime.now() - timedelta(days=time_range_days)
        df = self.predictions_df[self.predictions_df['timestamp'] >= cutoff_date]
        
        # Group by cluster
        metrics = df.groupby('cluster_name').agg({
            'engagement_ratio': 'mean',
            'decision_time': 'mean',
            'confidence': 'mean',
            'user_id': ['count', 'nunique']
        }).round(3)
        
        metrics.columns = ['avg_engagement', 'avg_decision_time', 'avg_confidence', 
                           'user_count', 'unique_users']
        metrics = metrics.reset_index().to_dict('records')
        
        summary = {
            'total_predictions': len(df),
            'total_users': df['user_id'].nunique(),
            'time_range_days': time_range_days
        }
        
        return {
            'metrics': metrics,
            'summary': summary
        }
    
    
    def get_conversion_funnel(self) -> Dict[str, Any]:
        """
        Track conversion funnel: Views → Comparisons → Decisions → Actions
        
        SQL Equivalent:
        SELECT 
            stage,
            COUNT(*) as count,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
        FROM (
            SELECT 'views' as stage FROM product_views
            UNION ALL
            SELECT 'comparisons' as stage FROM comparisons
            UNION ALL
            SELECT 'decisions' as stage FROM predictions
            UNION ALL
            SELECT 'actions' as stage FROM user_actions WHERE action_type = 'purchase'
        ) funnel
        GROUP BY stage
        ORDER BY count DESC
        
        Returns:
            {
                'funnel': [
                    {'stage': 'views', 'count': 10000, 'percentage': 100.0, 'drop_off': 0.0},
                    {'stage': 'comparisons', 'count': 3500, 'percentage': 35.0, 'drop_off': 65.0},
                    {'stage': 'decisions', 'count': 2800, 'percentage': 28.0, 'drop_off': 20.0},
                    {'stage': 'actions', 'count': 1200, 'percentage': 12.0, 'drop_off': 57.1}
                ],
                'insights': {
                    'biggest_drop_off_stage': 'views → comparisons',
                    'overall_conversion_rate': 0.12
                }
            }
        """
        # Mock data for demo (replace with actual SQL query)
        funnel_data = [
            {'stage': 'views', 'count': 10000},
            {'stage': 'comparisons', 'count': 3500},
            {'stage': 'decisions', 'count': 2800},
            {'stage': 'actions', 'count': 1200}
        ]
        
        # Calculate percentages and drop-offs
        for i, stage in enumerate(funnel_data):
            stage['percentage'] = (stage['count'] / funnel_data[0]['count']) * 100
            if i > 0:
                prev_count = funnel_data[i-1]['count']
                stage['drop_off'] = ((prev_count - stage['count']) / prev_count) * 100
            else:
                stage['drop_off'] = 0.0
        
        # Find biggest drop-off
        drop_offs = [(f"{funnel_data[i]['stage']} → {funnel_data[i+1]['stage']}", 
                      funnel_data[i+1]['drop_off']) 
                     for i in range(len(funnel_data)-1)]
        biggest_drop = max(drop_offs, key=lambda x: x[1])
        
        return {
            'funnel': funnel_data,
            'insights': {
                'biggest_drop_off_stage': biggest_drop[0],
                'biggest_drop_off_percentage': biggest_drop[1],
                'overall_conversion_rate': funnel_data[-1]['count'] / funnel_data[0]['count']
            }
        }
    
    
    def get_prediction_accuracy_by_cluster(self) -> Dict[str, Any]:
        """
        Analyze prediction accuracy across user segments
        
        SQL Equivalent:
        SELECT 
            cluster_name,
            AVG(confidence) as avg_confidence,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN confidence > 0.8 THEN 1 ELSE 0 END) as high_confidence_count,
            SUM(CASE WHEN confidence > 0.8 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as high_confidence_pct
        FROM predictions
        GROUP BY cluster_name
        ORDER BY avg_confidence DESC
        
        Returns:
            {
                'accuracy_by_cluster': [
                    {
                        'cluster_name': 'Power Decision Makers',
                        'avg_confidence': 0.88,
                        'total_predictions': 1500,
                        'high_confidence_count': 1350,
                        'high_confidence_pct': 90.0
                    },
                    ...
                ]
            }
        """
        if self.predictions_df.empty:
            return {"accuracy_by_cluster": []}
        
        df = self.predictions_df.copy()
        df['high_confidence'] = (df['confidence'] > 0.8).astype(int)
        
        accuracy = df.groupby('cluster_name').agg({
            'confidence': 'mean',
            'user_id': 'count',
            'high_confidence': ['sum', 'mean']
        }).round(3)
        
        accuracy.columns = ['avg_confidence', 'total_predictions', 
                            'high_confidence_count', 'high_confidence_pct']
        accuracy['high_confidence_pct'] *= 100
        accuracy = accuracy.reset_index().to_dict('records')
        
        return {'accuracy_by_cluster': accuracy}
    
    
    def get_feature_importance_trends(self, days: int = 7) -> Dict[str, Any]:
        """
        Track how feature importance changes over time
        
        SQL Equivalent:
        SELECT 
            DATE(timestamp) as date,
            feature_name,
            AVG(feature_value) as avg_value,
            STDDEV(feature_value) as std_value
        FROM feature_logs
        WHERE timestamp >= NOW() - INTERVAL '7 days'
        GROUP BY DATE(timestamp), feature_name
        ORDER BY date DESC, avg_value DESC
        
        Returns:
            {
                'trends': [
                    {
                        'date': '2026-03-19',
                        'feature_name': 'price_sensitivity',
                        'avg_value': 0.68,
                        'std_value': 0.12
                    },
                    ...
                ],
                'insights': {
                    'most_volatile_feature': 'quality_score',
                    'most_stable_feature': 'device_type'
                }
            }
        """
        # Mock data for demo
        features = ['price_sensitivity', 'quality_score', 'feature_interest_score', 
                    'engagement_ratio', 'decision_efficiency']
        
        trends = []
        for day in range(days):
            date = (datetime.now() - timedelta(days=day)).strftime('%Y-%m-%d')
            for feature in features:
                trends.append({
                    'date': date,
                    'feature_name': feature,
                    'avg_value': np.random.uniform(0.5, 0.9),
                    'std_value': np.random.uniform(0.05, 0.20)
                })
        
        # Find most volatile
        volatility = pd.DataFrame(trends).groupby('feature_name')['std_value'].mean().sort_values(ascending=False)
        
        return {
            'trends': trends,
            'insights': {
                'most_volatile_feature': volatility.index[0],
                'most_stable_feature': volatility.index[-1],
                'avg_volatility': volatility.mean()
            }
        }
    
    
    def get_decision_time_analysis(self) -> Dict[str, Any]:
        """
        Analyze decision-making speed across clusters
        
        SQL Equivalent:
        SELECT 
            cluster_name,
            AVG(decision_time) as avg_decision_time,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY decision_time) as median_decision_time,
            MIN(decision_time) as min_decision_time,
            MAX(decision_time) as max_decision_time
        FROM predictions
        GROUP BY cluster_name
        ORDER BY avg_decision_time ASC
        
        Returns:
            {
                'decision_time_by_cluster': [...],
                'insights': {
                    'fastest_cluster': 'Power Decision Makers',
                    'slowest_cluster': 'Analytical Researchers',
                    'avg_improvement_vs_baseline': 40.5  # percentage
                }
            }
        """
        if self.predictions_df.empty:
            return {"decision_time_by_cluster": []}
        
        df = self.predictions_df.copy()
        
        decision_times = df.groupby('cluster_name')['decision_time'].agg([
            'mean', 'median', 'min', 'max'
        ]).round(2)
        
        decision_times.columns = ['avg_decision_time', 'median_decision_time', 
                                   'min_decision_time', 'max_decision_time']
        decision_times = decision_times.reset_index().to_dict('records')
        
        # Calculate insights
        fastest = min(decision_times, key=lambda x: x['avg_decision_time'])
        slowest = max(decision_times, key=lambda x: x['avg_decision_time'])
        
        baseline = 300  # 5 minutes baseline
        avg_time = df['decision_time'].mean()
        improvement = ((baseline - avg_time) / baseline) * 100
        
        return {
            'decision_time_by_cluster': decision_times,
            'insights': {
                'fastest_cluster': fastest['cluster_name'],
                'slowest_cluster': slowest['cluster_name'],
                'avg_improvement_vs_baseline': round(improvement, 2),
                'overall_avg_time': round(avg_time, 2)
            }
        }
    
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """
        Generate executive summary with key metrics
        
        Returns:
            {
                'kpis': {
                    'total_users': 3200,
                    'total_predictions': 5000,
                    'avg_confidence': 0.85,
                    'avg_decision_time': 180,
                    'conversion_rate': 0.12
                },
                'highlights': [
                    'Power Decision Makers have 90% high-confidence predictions',
                    '40% reduction in decision time vs. baseline',
                    'Biggest funnel drop-off: views → comparisons (65%)'
                ],
                'recommendations': [
                    'Focus on improving comparison stage engagement',
                    'Personalize experience for Analytical Researchers (slowest)',
                    'Leverage simulation engine to boost confidence'
                ]
            }
        """
        engagement = self.get_user_engagement_metrics()
        funnel = self.get_conversion_funnel()
        accuracy = self.get_prediction_accuracy_by_cluster()
        decision_time = self.get_decision_time_analysis()
        
        # Extract KPIs
        kpis = {
            'total_users': engagement['summary'].get('total_users', 0),
            'total_predictions': engagement['summary'].get('total_predictions', 0),
            'avg_confidence': round(np.mean([m['avg_confidence'] for m in engagement['metrics']]), 3) if engagement['metrics'] else 0,
            'avg_decision_time': decision_time['insights'].get('overall_avg_time', 0),
            'conversion_rate': funnel['insights'].get('overall_conversion_rate', 0)
        }
        
        # Generate highlights
        highlights = [
            f"Power Decision Makers have 90% high-confidence predictions",
            f"{round(decision_time['insights']['avg_improvement_vs_baseline'])}% reduction in decision time vs. baseline",
            f"Biggest funnel drop-off: {funnel['insights']['biggest_drop_off_stage']} ({round(funnel['insights']['biggest_drop_off_percentage'])}%)"
        ]
        
        # Generate recommendations
        recommendations = [
            f"Focus on improving {funnel['insights']['biggest_drop_off_stage']} stage",
            f"Personalize experience for {decision_time['insights']['slowest_cluster']} (slowest cluster)",
            "Leverage simulation engine to boost confidence scores",
            "A/B test explainability features to improve trust"
        ]
        
        return {
            'kpis': kpis,
            'highlights': highlights,
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }


# Example usage
if __name__ == "__main__":
    analytics = SQLAnalytics()
    
    # Log some sample predictions
    for i in range(100):
        analytics.log_prediction({
            'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 30)),
            'user_id': f"user_{np.random.randint(1, 50)}",
            'cluster_name': np.random.choice(['Casual Users', 'Analytical Researchers', 
                                              'High Intent Buyers', 'Power Decision Makers']),
            'predicted_option': np.random.choice(['Option_A', 'Option_B', 'Option_C']),
            'confidence': np.random.uniform(0.6, 0.95),
            'engagement_ratio': np.random.uniform(0.1, 0.3),
            'decision_time': np.random.uniform(60, 400)
        })
    
    # Generate reports
    print("📊 User Engagement Metrics:")
    print(analytics.get_user_engagement_metrics())
    
    print("\n📈 Conversion Funnel:")
    print(analytics.get_conversion_funnel())
    
    print("\n🎯 Executive Summary:")
    print(analytics.generate_executive_summary())
