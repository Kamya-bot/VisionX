"""
VisionX - Model Versioning API Routes
Track and manage ML model versions
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime
import logging

from services.model_versioning import get_model_registry, ModelVersion

router = APIRouter(prefix="/models", tags=["Model Versioning"])
logger = logging.getLogger(__name__)


@router.get("/versions")
async def list_model_versions(
    model_type: Optional[str] = Query(default=None, pattern="^(clustering|prediction)$"),
    limit: int = Query(default=10, ge=1, le=50)
):
    try:
        registry = get_model_registry()
        versions = registry.list_versions(model_type=model_type, limit=limit)
        
        return {
            'count': len(versions),
            'versions': [
                {
                    'version_id': v.version_id,
                    'model_type': v.model_type,
                    'created_at': v.created_at.isoformat(),
                    'features_count': len(v.features),
                    'metrics': v.metrics,
                    'hyperparameters': v.hyperparameters
                }
                for v in versions
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to list versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/versions/{version_id}")
async def get_model_version_detail(version_id: str):
    try:
        registry = get_model_registry()
        version = registry.get_version(version_id)
        
        if not version:
            raise HTTPException(
                status_code=404,
                detail=f"Model version '{version_id}' not found"
            )
        
        return {
            'version_id': version.version_id,
            'model_type': version.model_type,
            'created_at': version.created_at.isoformat(),
            'model_path': version.model_path,
            'features': version.features,
            'feature_count': len(version.features),
            'metrics': version.metrics,
            'hyperparameters': version.hyperparameters,
            'dataset_info': version.dataset_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get version detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest")
async def get_latest_model_version(
    model_type: str = Query(..., pattern="^(clustering|prediction)$")
):
    try:
        registry = get_model_registry()
        version = registry.get_latest_version(model_type)
        
        if not version:
            return {
                'message': f'No {model_type} models found in registry',
                'recommendation': 'Train a model first using training/train_models.py'
            }
        
        return {
            'version_id': version.version_id,
            'model_type': version.model_type,
            'created_at': version.created_at.isoformat(),
            'features_count': len(version.features),
            'metrics': version.metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to get latest version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/best")
async def get_best_model_version(
    model_type: str = Query(..., pattern="^(clustering|prediction)$"),
    metric: str = Query(default='accuracy')
):
    try:
        registry = get_model_registry()
        best_version = registry.get_best_version(model_type, metric)
        
        if not best_version:
            return {
                'message': f'No {model_type} models found',
                'recommendation': 'Train models first'
            }
        
        return {
            'version_id': best_version.version_id,
            'model_type': best_version.model_type,
            'created_at': best_version.created_at.isoformat(),
            'features_count': len(best_version.features),
            'metrics': best_version.metrics,
            'optimized_metric': metric,
            'metric_value': best_version.metrics.get(metric, 0.0)
        }
        
    except Exception as e:
        logger.error(f"Failed to get best version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare")
async def compare_model_versions(
    version_1: str = Query(..., description="First version ID"),
    version_2: str = Query(..., description="Second version ID")
):
    try:
        registry = get_model_registry()
        comparison = registry.compare_versions(version_1, version_2)
        
        metric_improvements = []
        metric_regressions = []
        
        for metric, data in comparison['metric_comparison'].items():
            if data['diff'] > 0:
                metric_improvements.append({
                    'metric': metric,
                    'improvement': f"+{data['percent_change']:.2f}%"
                })
            elif data['diff'] < 0:
                metric_regressions.append({
                    'metric': metric,
                    'regression': f"{data['percent_change']:.2f}%"
                })
        
        comparison['summary'] = {
            'improvements': metric_improvements,
            'regressions': metric_regressions,
            'overall_better': version_2 if len(metric_improvements) > len(metric_regressions) else version_1
        }
        
        return comparison
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to compare versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/history")
async def get_metrics_history(
    model_type: str = Query(..., pattern="^(clustering|prediction)$"),
    metric: str = Query(default='accuracy')
):
    try:
        registry = get_model_registry()
        versions = registry.list_versions(model_type=model_type, limit=50)
        
        if not versions:
            return {
                'message': f'No {model_type} models found',
                'data': []
            }
        
        history = []
        for version in reversed(versions):
            metric_value = version.metrics.get(metric)
            if metric_value is not None:
                history.append({
                    'version_id': version.version_id,
                    'date': version.created_at.isoformat(),
                    'value': metric_value
                })
        
        return {
            'model_type': model_type,
            'metric': metric,
            'data_points': len(history),
            'history': history
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics history: {e}")
        raise HTTPException(status_code=500, detail=str(e))