"""
VisionX API - Analytics Endpoints

Expose analytics capabilities: KPIs, A/B testing, business insights
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List
from datetime import datetime
import logging
from sqlalchemy.orm import Session

from analytics.sql_analytics import SQLAnalytics
from analytics.kpi_tracker import KPITracker
from analytics.ab_testing import ABTestFramework
from schemas.response_models import (
    AnalyticsResponse,
    KPIResponse,
    ABTestResponse
)
from database import get_db
from crud import (
    get_decision_success_rate,
    get_average_satisfaction,
    get_active_users_count,
    get_predictions_count,
    get_average_confidence
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize analytics components (with mock fallback)
sql_analytics = SQLAnalytics()
kpi_tracker = KPITracker()
ab_test_framework = ABTestFramework()


# ========== SQL Analytics Endpoints ==========

@router.get("/analytics/engagement", response_model=AnalyticsResponse)
async def get_engagement_metrics(time_range_days: int = Query(30, ge=1, le=365)):
    """
    Get user engagement metrics by cluster
    
    Query Params:
    - time_range_days: Number of days to analyze (default: 30)
    
    Returns:
    - Engagement metrics grouped by user cluster
    - Average session time, clicks, scroll depth
    - User counts per cluster
    """
    try:
        metrics = sql_analytics.get_user_engagement_metrics(time_range_days)
        
        return {
            "status": "success",
            "data": metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching engagement metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/funnel", response_model=AnalyticsResponse)
async def get_conversion_funnel():
    """
    Get conversion funnel analysis
    
    Returns:
    - Funnel stages: views → comparisons → decisions → actions
    - Drop-off rates at each stage
    - Biggest bottleneck identification
    """
    try:
        funnel = sql_analytics.get_conversion_funnel()
        
        return {
            "status": "success",
            "data": funnel,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching funnel: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/accuracy", response_model=AnalyticsResponse)
async def get_prediction_accuracy_by_cluster():
    """
    Get prediction accuracy analysis by user cluster
    
    Returns:
    - Average confidence per cluster
    - High-confidence prediction rates
    - Total predictions per cluster
    """
    try:
        accuracy = sql_analytics.get_prediction_accuracy_by_cluster()
        
        return {
            "status": "success",
            "data": accuracy,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching accuracy metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/feature-trends", response_model=AnalyticsResponse)
async def get_feature_importance_trends(days: int = Query(7, ge=1, le=30)):
    """
    Get feature importance trends over time
    
    Query Params:
    - days: Number of days to analyze (default: 7)
    
    Returns:
    - Feature importance time series
    - Most volatile and stable features
    """
    try:
        trends = sql_analytics.get_feature_importance_trends(days)
        
        return {
            "status": "success",
            "data": trends,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching feature trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/decision-time", response_model=AnalyticsResponse)
async def get_decision_time_analysis():
    """
    Get decision-making speed analysis
    
    Returns:
    - Average/median decision time per cluster
    - Fastest and slowest clusters
    - Improvement vs. baseline
    """
    try:
        analysis = sql_analytics.get_decision_time_analysis()
        
        return {
            "status": "success",
            "data": analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching decision time analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/executive-summary", response_model=AnalyticsResponse)
async def get_executive_summary():
    """
    Get executive summary with key insights
    
    Returns:
    - Top KPIs (users, predictions, confidence, conversion)
    - Key highlights
    - Actionable recommendations
    """
    try:
        summary = sql_analytics.generate_executive_summary()
        
        return {
            "status": "success",
            "data": summary,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error generating executive summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== KPI Tracking Endpoints ==========

@router.get("/analytics/kpis", response_model=KPIResponse)
async def get_all_kpis(db: Session = Depends(get_db)):
    """
    Get all current KPIs grouped by category - REAL DATABASE VERSION
    
    Returns:
    - User engagement KPIs (from real DB)
    - ML performance KPIs (from real DB)
    - Decision quality KPIs (from real DB)
    - Business KPIs
    - Technical KPIs
    - Feature usage KPIs
    - Summary with health score
    """
    try:
        # ✅ TRY REAL DATABASE FIRST
        try:
            real_kpis = {
                "decision_success_rate": get_decision_success_rate(db, days=30),
                "avg_satisfaction": get_average_satisfaction(db, days=30),
                "active_users": get_active_users_count(db, days=30),
                "total_predictions": get_predictions_count(db, days=1),
                "avg_confidence": get_average_confidence(db, days=7),
                "source": "real_database",
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("✅ Returning KPIs from REAL DATABASE")
            
            return {
                "status": "success",
                "data": real_kpis,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as db_error:
            logger.warning(f"⚠️  Real DB query failed, using mock KPIs: {str(db_error)}")
            # Fall back to mock data
            kpis = kpi_tracker.get_all_kpis()
            kpis["source"] = "mock_fallback"
            
            return {
                "status": "success",
                "data": kpis,
                "timestamp": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Error fetching KPIs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/kpis/{kpi_name}", response_model=KPIResponse)
async def get_single_kpi(kpi_name: str):
    """
    Get single KPI details
    
    Path Params:
    - kpi_name: Name of KPI (e.g., 'prediction_accuracy')
    
    Returns:
    - Current value
    - Status (excellent/good/warning/critical)
    - Target and critical thresholds
    """
    try:
        kpi = kpi_tracker.get_kpi(kpi_name)
        
        if "error" in kpi:
            raise HTTPException(status_code=404, detail=kpi["error"])
        
        return {
            "status": "success",
            "data": kpi,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching KPI {kpi_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/kpis/{kpi_name}/trend", response_model=KPIResponse)
async def get_kpi_trend(kpi_name: str, days: int = Query(7, ge=1, le=30)):
    """
    Get KPI trend analysis
    
    Path Params:
    - kpi_name: Name of KPI
    
    Query Params:
    - days: Number of days to analyze (default: 7)
    
    Returns:
    - Trend direction (improving/declining/stable)
    - Change percentage
    - Time series data
    """
    try:
        trend = kpi_tracker.get_kpi_trend(kpi_name, days)
        
        if "error" in trend:
            raise HTTPException(status_code=404, detail=trend["error"])
        
        return {
            "status": "success",
            "data": trend,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching trend for {kpi_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/kpis/alerts", response_model=KPIResponse)
async def get_kpi_alerts():
    """
    Get KPIs in warning or critical status
    
    Returns:
    - List of KPIs requiring attention
    - Current vs. target values
    - Alert messages
    """
    try:
        alerts = kpi_tracker.get_alerts()
        
        return {
            "status": "success",
            "data": {"alerts": alerts},
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/kpi-report", response_model=KPIResponse)
async def get_kpi_report(days: int = Query(7, ge=1, le=30)):
    """
    Get comprehensive KPI report
    
    Query Params:
    - days: Time range for trends (default: 7)
    
    Returns:
    - Summary with health score
    - KPIs by category
    - Trends analysis
    - Alerts
    - Recommendations
    """
    try:
        report = kpi_tracker.generate_kpi_report(days)
        
        return {
            "status": "success",
            "data": report,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error generating KPI report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== A/B Testing Endpoints ==========

@router.post("/analytics/experiments", response_model=ABTestResponse)
async def create_ab_experiment(experiment_data: dict):
    """
    Create new A/B test experiment
    
    Request Body:
    {
        "name": "experiment_name",
        "description": "What's being tested",
        "variants": [
            {"name": "control", "description": "Current version", "weight": 0.5},
            {"name": "treatment", "description": "New version", "weight": 0.5}
        ],
        "metric_name": "conversion_rate",
        "metric_type": "binary",  # or "continuous"
        "traffic_allocation": 1.0
    }
    
    Returns:
    - Experiment metadata
    - Experiment ID
    """
    try:
        experiment = ab_test_framework.create_experiment(**experiment_data)
        
        return {
            "status": "success",
            "data": experiment,
            "timestamp": datetime.now().isoformat()
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analytics/experiments/{experiment_id}/start", response_model=ABTestResponse)
async def start_ab_experiment(experiment_id: str):
    """
    Start running an A/B test experiment
    
    Path Params:
    - experiment_id: Experiment identifier
    
    Returns:
    - Updated experiment status
    """
    try:
        experiment = ab_test_framework.start_experiment(experiment_id)
        
        return {
            "status": "success",
            "data": experiment,
            "timestamp": datetime.now().isoformat()
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/experiments/{experiment_id}/assign/{user_id}", response_model=ABTestResponse)
async def assign_user_to_variant(experiment_id: str, user_id: str):
    """
    Assign user to experiment variant
    
    Path Params:
    - experiment_id: Experiment identifier
    - user_id: User identifier
    
    Returns:
    - Assigned variant name
    """
    try:
        variant = ab_test_framework.assign_variant(experiment_id, user_id)
        
        return {
            "status": "success",
            "data": {"variant": variant, "user_id": user_id},
            "timestamp": datetime.now().isoformat()
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error assigning variant: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analytics/experiments/{experiment_id}/track", response_model=ABTestResponse)
async def track_experiment_metric(experiment_id: str, metric_data: dict):
    """
    Track metric for user in experiment
    
    Path Params:
    - experiment_id: Experiment identifier
    
    Request Body:
    {
        "user_id": "user_123",
        "metric_value": 0.85
    }
    
    Returns:
    - Confirmation of metric tracked
    """
    try:
        user_id = metric_data["user_id"]
        metric_value = metric_data["metric_value"]
        
        ab_test_framework.track_metric(experiment_id, user_id, metric_value)
        
        return {
            "status": "success",
            "data": {"message": "Metric tracked successfully"},
            "timestamp": datetime.now().isoformat()
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
    except Exception as e:
        logger.error(f"Error tracking metric: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/experiments/{experiment_id}/results", response_model=ABTestResponse)
async def get_experiment_results(experiment_id: str):
    """
    Analyze A/B test results with statistical significance
    
    Path Params:
    - experiment_id: Experiment identifier
    
    Returns:
    - Variant statistics (sample size, mean, std)
    - Pairwise comparisons (p-value, lift, significance)
    - Winner recommendation
    """
    try:
        results = ab_test_framework.analyze_results(experiment_id)
        
        if "error" in results:
            raise HTTPException(status_code=404, detail=results["error"])
        
        return {
            "status": "success",
            "data": results,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing experiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/experiments", response_model=ABTestResponse)
async def list_ab_experiments(status: Optional[str] = None):
    """
    List all A/B test experiments
    
    Query Params:
    - status: Filter by status (running, completed, etc.)
    
    Returns:
    - List of experiments with metadata
    """
    try:
        experiments = ab_test_framework.list_experiments(status)
        
        return {
            "status": "success",
            "data": {"experiments": experiments, "count": len(experiments)},
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error listing experiments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/sample-size-calculator", response_model=ABTestResponse)
async def calculate_required_sample_size(
    baseline_conversion: float = Query(..., ge=0.0, le=1.0),
    mde: float = Query(..., ge=0.01, le=1.0),
    alpha: float = Query(0.05, ge=0.01, le=0.10),
    power: float = Query(0.80, ge=0.70, le=0.95)
):
    """
    Calculate required sample size for A/B test
    
    Query Params:
    - baseline_conversion: Current conversion rate (0.0 to 1.0)
    - mde: Minimum Detectable Effect (e.g., 0.05 for 5% relative lift)
    - alpha: Significance level (default: 0.05)
    - power: Statistical power (default: 0.80)
    
    Returns:
    - Required sample size per variant
    """
    try:
        sample_size = ab_test_framework.compute_required_sample_size(
            baseline_conversion, mde, alpha, power
        )
        
        return {
            "status": "success",
            "data": {
                "required_sample_size_per_variant": sample_size,
                "total_sample_size": sample_size * 2,  # Assuming 2 variants
                "parameters": {
                    "baseline_conversion": baseline_conversion,
                    "minimum_detectable_effect": mde,
                    "alpha": alpha,
                    "power": power
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error calculating sample size: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
