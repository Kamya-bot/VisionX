"""
VisionX - Model Versioning & Experiment Tracking
Track model versions, performance metrics, and enable A/B testing
"""

import json
import joblib
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from sqlalchemy.orm import Session
import logging

logger = logging.getLogger(__name__)


class ModelVersion:
    """
    Represents a specific version of an ML model
    
    Tracks:
    - Model binary (joblib file)
    - Metadata (features, hyperparameters, training date)
    - Performance metrics (accuracy, F1, etc.)
    - Training dataset info
    """
    
    def __init__(
        self,
        version_id: str,
        model_type: str,
        model_path: str,
        features: List[str],
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        created_at: datetime,
        dataset_info: Optional[Dict] = None
    ):
        self.version_id = version_id
        self.model_type = model_type  # 'clustering' or 'prediction'
        self.model_path = model_path
        self.features = features
        self.metrics = metrics
        self.hyperparameters = hyperparameters
        self.created_at = created_at
        self.dataset_info = dataset_info or {}
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'version_id': self.version_id,
            'model_type': self.model_type,
            'model_path': self.model_path,
            'features': self.features,
            'metrics': self.metrics,
            'hyperparameters': self.hyperparameters,
            'created_at': self.created_at.isoformat(),
            'dataset_info': self.dataset_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        """Create from dictionary"""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class ModelRegistry:
    """
    Central registry for all model versions
    
    Features:
    - Register new model versions
    - Track performance over time
    - Enable model rollback
    - Support A/B testing (multiple active versions)
    """
    
    def __init__(self, base_path: str = "backend/models"):
        self.base_path = Path(base_path)
        self.registry_file = self.base_path / "model_registry.json"
        self.versions: Dict[str, List[ModelVersion]] = {}
        
        # Load existing registry
        self._load_registry()
    
    def register_model(
        self,
        model_type: str,
        model_obj: Any,
        features: List[str],
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        dataset_info: Optional[Dict] = None
    ) -> ModelVersion:
        """
        Register a new model version
        
        Args:
            model_type: 'clustering' or 'prediction'
            model_obj: The actual sklearn/xgboost model
            features: List of feature names
            metrics: Performance metrics (accuracy, f1_score, etc.)
            hyperparameters: Model hyperparameters
            dataset_info: Info about training dataset
            
        Returns:
            ModelVersion object
        """
        # Generate version ID (timestamp + hash)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        model_hash = self._generate_model_hash(model_obj)
        version_id = f"{model_type}_v{timestamp}_{model_hash[:8]}"
        
        # Save model to disk
        model_filename = f"{version_id}.joblib"
        model_path = str(self.base_path / model_filename)
        joblib.dump(model_obj, model_path)
        
        # Create version object
        version = ModelVersion(
            version_id=version_id,
            model_type=model_type,
            model_path=model_path,
            features=features,
            metrics=metrics,
            hyperparameters=hyperparameters,
            created_at=datetime.utcnow(),
            dataset_info=dataset_info
        )
        
        # Add to registry
        if model_type not in self.versions:
            self.versions[model_type] = []
        
        self.versions[model_type].append(version)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"✅ Registered model version: {version_id}")
        logger.info(f"   Metrics: {metrics}")
        logger.info(f"   Features: {len(features)}")
        
        return version
    
    def get_latest_version(self, model_type: str) -> Optional[ModelVersion]:
        """Get the most recent version of a model type"""
        if model_type not in self.versions or len(self.versions[model_type]) == 0:
            return None
        
        # Sort by creation date, return latest
        versions = sorted(
            self.versions[model_type],
            key=lambda v: v.created_at,
            reverse=True
        )
        return versions[0]
    
    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a specific model version by ID"""
        for model_type, versions in self.versions.items():
            for version in versions:
                if version.version_id == version_id:
                    return version
        return None
    
    def list_versions(
        self,
        model_type: Optional[str] = None,
        limit: int = 10
    ) -> List[ModelVersion]:
        """
        List model versions
        
        Args:
            model_type: Filter by model type (optional)
            limit: Max number of versions to return
            
        Returns:
            List of ModelVersion objects, sorted by date (newest first)
        """
        all_versions = []
        
        if model_type:
            all_versions = self.versions.get(model_type, [])
        else:
            # All model types
            for versions in self.versions.values():
                all_versions.extend(versions)
        
        # Sort by creation date
        all_versions = sorted(
            all_versions,
            key=lambda v: v.created_at,
            reverse=True
        )
        
        return all_versions[:limit]
    
    def compare_versions(
        self,
        version_id_1: str,
        version_id_2: str
    ) -> Dict:
        """
        Compare two model versions
        
        Returns:
            Dict with comparison of metrics, features, hyperparameters
        """
        v1 = self.get_version(version_id_1)
        v2 = self.get_version(version_id_2)
        
        if not v1 or not v2:
            raise ValueError("Version not found")
        
        if v1.model_type != v2.model_type:
            raise ValueError("Cannot compare different model types")
        
        # Compare metrics
        metric_comparison = {}
        all_metrics = set(v1.metrics.keys()) | set(v2.metrics.keys())
        
        for metric in all_metrics:
            val1 = v1.metrics.get(metric, 0.0)
            val2 = v2.metrics.get(metric, 0.0)
            metric_comparison[metric] = {
                'version_1': val1,
                'version_2': val2,
                'diff': val2 - val1,
                'percent_change': ((val2 - val1) / val1 * 100) if val1 != 0 else 0.0
            }
        
        # Compare features
        features_added = list(set(v2.features) - set(v1.features))
        features_removed = list(set(v1.features) - set(v2.features))
        
        return {
            'version_1': {
                'id': v1.version_id,
                'created_at': v1.created_at.isoformat(),
                'metrics': v1.metrics,
                'feature_count': len(v1.features)
            },
            'version_2': {
                'id': v2.version_id,
                'created_at': v2.created_at.isoformat(),
                'metrics': v2.metrics,
                'feature_count': len(v2.features)
            },
            'metric_comparison': metric_comparison,
            'features': {
                'added': features_added,
                'removed': features_removed,
                'unchanged_count': len(set(v1.features) & set(v2.features))
            }
        }
    
    def get_best_version(
        self,
        model_type: str,
        metric: str = 'accuracy'
    ) -> Optional[ModelVersion]:
        """
        Get the best-performing version based on a metric
        
        Args:
            model_type: 'clustering' or 'prediction'
            metric: Metric to optimize (e.g., 'accuracy', 'f1_score')
            
        Returns:
            ModelVersion with highest metric value
        """
        versions = self.versions.get(model_type, [])
        
        if not versions:
            return None
        
        # Find version with max metric
        best_version = max(
            versions,
            key=lambda v: v.metrics.get(metric, 0.0)
        )
        
        return best_version
    
    def load_model(self, version_id: str) -> Any:
        """Load a specific model version from disk"""
        version = self.get_version(version_id)
        
        if not version:
            raise ValueError(f"Version {version_id} not found")
        
        return joblib.load(version.model_path)
    
    def _generate_model_hash(self, model_obj: Any) -> str:
        """Generate hash of model for versioning"""
        # Serialize model to bytes
        import pickle
        model_bytes = pickle.dumps(model_obj)
        
        # Generate hash
        return hashlib.md5(model_bytes).hexdigest()
    
    def _load_registry(self):
        """Load registry from JSON file"""
        if not self.registry_file.exists():
            logger.info("No existing model registry found. Creating new one.")
            return
        
        try:
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct versions
            for model_type, versions_data in data.items():
                self.versions[model_type] = [
                    ModelVersion.from_dict(v) for v in versions_data
                ]
            
            logger.info(f"✅ Loaded model registry: {sum(len(v) for v in self.versions.values())} versions")
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
    
    def _save_registry(self):
        """Save registry to JSON file"""
        try:
            # Ensure directory exists
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Convert to dict
            data = {}
            for model_type, versions in self.versions.items():
                data[model_type] = [v.to_dict() for v in versions]
            
            # Save to JSON
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"✅ Saved model registry to {self.registry_file}")
            
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")


# Global registry instance
model_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance"""
    return model_registry
