"""
SHAP-Based Model Explainability Module
Provides interpretable explanations for ML predictions

Why this matters:
- Shows WHY the model made a prediction
- Critical for production ML systems
- Companies like Google & Microsoft require this
"""

import shap
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based explainability for XGBoost models
    
    Features:
    - Feature importance calculation
    - Top feature identification
    - Human-readable explanations
    - SHAP value computation
    """
    
    def __init__(self, model_path: str, feature_names: List[str]):
        """
        Initialize SHAP explainer
        
        Args:
            model_path: Path to trained model file
            feature_names: List of feature names in order
        """
        try:
            self.model = joblib.load(model_path)
            self.feature_names = feature_names
            self.explainer = shap.TreeExplainer(self.model)
            logger.info(f"✅ SHAP Explainer initialized with {len(feature_names)} features")
        except Exception as e:
            logger.error(f"❌ Failed to initialize SHAP explainer: {str(e)}")
            raise
    
    def explain_prediction(self, input_data: np.ndarray) -> Dict[str, Any]:
        """
        Generate SHAP-based explanation for prediction
        
        Args:
            input_data: Feature vector (1D or 2D numpy array)
            
        Returns:
            Dictionary with SHAP values, feature importance, and top features
        """
        try:
            # Ensure input is 2D
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(input_data)
            
            # Handle multi-class output (take first sample if batch)
            if isinstance(shap_values, list):
                # Multi-class: average across classes
                shap_values = np.mean(shap_values, axis=0)
            
            if shap_values.ndim > 1:
                shap_values = shap_values[0]  # Take first sample
            
            # Calculate feature importance (absolute mean)
            feature_importance = np.abs(shap_values)
            
            # Get top features
            top_features = self.get_top_features(feature_importance, k=5)
            
            # Generate human-readable explanation
            explanation = self.generate_explanation(top_features)
            
            return {
                "shap_values": shap_values.tolist(),
                "feature_importance": dict(zip(self.feature_names, feature_importance.tolist())),
                "top_features": top_features,
                "explanation": explanation,
                "base_value": float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.0
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to explain prediction: {str(e)}")
            return {
                "error": str(e),
                "shap_values": [],
                "feature_importance": {},
                "top_features": [],
                "explanation": "Failed to generate explanation"
            }
    
    def get_top_features(self, importance: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top K most important features
        
        Args:
            importance: Feature importance scores
            k: Number of top features to return
            
        Returns:
            List of (feature_name, importance_score) tuples
        """
        # Create pairs of (feature_name, importance)
        pairs = list(zip(self.feature_names, importance))
        
        # Sort by importance (descending)
        pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K
        return [(name, float(score)) for name, score in pairs[:k]]
    
    def generate_explanation(self, top_features: List[Tuple[str, float]]) -> str:
        """
        Generate human-readable explanation from top features
        
        Args:
            top_features: List of (feature_name, importance) tuples
            
        Returns:
            Human-readable explanation string
        """
        if not top_features:
            return "Unable to generate explanation"
        
        # Get top 3 features
        top_3 = top_features[:3]
        
        explanations = []
        
        for feature_name, importance in top_3:
            # Map feature names to human-readable descriptions
            feature_descriptions = {
                "price_sensitivity": "price consciousness",
                "risk_tolerance": "willingness to take risks",
                "brand_loyalty": "brand preference strength",
                "decision_speed": "decision-making speed",
                "research_time": "research thoroughness",
                "quality_score": "quality preference",
                "feature_count": "feature importance",
                "engagement_score": "engagement level",
                "purchase_intent_score": "purchase likelihood",
                "comparison_count": "comparison behavior",
                "session_time": "time invested",
                "clicks": "interaction frequency"
            }
            
            description = feature_descriptions.get(feature_name, feature_name)
            
            if importance > 0.3:
                strength = "strongly"
            elif importance > 0.15:
                strength = "moderately"
            else:
                strength = "slightly"
            
            explanations.append(f"{strength} influenced by {description}")
        
        # Combine into coherent sentence
        if len(explanations) == 1:
            return f"This prediction is {explanations[0]}."
        elif len(explanations) == 2:
            return f"This prediction is {explanations[0]} and {explanations[1]}."
        else:
            return f"This prediction is {', '.join(explanations[:-1])}, and {explanations[-1]}."
    
    def explain_batch(self, input_data: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Explain multiple predictions at once
        
        Args:
            input_data: 2D numpy array of feature vectors
            top_k: Number of top features to include
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        
        for i in range(input_data.shape[0]):
            sample = input_data[i:i+1]
            explanation = self.explain_prediction(sample)
            explanation['sample_id'] = i
            explanations.append(explanation)
        
        return explanations
    
    def get_feature_contributions(self, input_data: np.ndarray) -> pd.DataFrame:
        """
        Get detailed feature contributions for visualization
        
        Args:
            input_data: Feature vector
            
        Returns:
            DataFrame with feature names, values, and contributions
        """
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        shap_values = self.explainer.shap_values(input_data)
        
        if isinstance(shap_values, list):
            shap_values = np.mean(shap_values, axis=0)
        
        if shap_values.ndim > 1:
            shap_values = shap_values[0]
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'value': input_data[0],
            'shap_contribution': shap_values,
            'abs_contribution': np.abs(shap_values)
        })
        
        df = df.sort_values('abs_contribution', ascending=False)
        
        return df


def create_explainer_for_model(model_path: str, feature_names: List[str]) -> SHAPExplainer:
    """
    Factory function to create SHAP explainer
    
    Args:
        model_path: Path to model file
        feature_names: List of feature names
        
    Returns:
        Initialized SHAPExplainer instance
    """
    return SHAPExplainer(model_path, feature_names)


# Example usage
if __name__ == "__main__":
    # Example: Create explainer
    feature_names = [
        "session_time", "clicks", "scroll_depth", "comparison_count",
        "product_views", "engagement_score", "purchase_intent_score"
    ]
    
    # Mock data for testing
    sample_input = np.array([450, 18, 0.75, 4, 10, 0.78, 0.85])
    
    print("🔍 SHAP Explainability Module")
    print("="*60)
    print("This module provides:")
    print("  ✅ Feature importance calculation")
    print("  ✅ Top feature identification")
    print("  ✅ Human-readable explanations")
    print("  ✅ SHAP value computation")
    print("\n📊 Example Explanation:")
    print("  'This prediction is strongly influenced by purchase likelihood,")
    print("   moderately influenced by engagement level, and slightly")
    print("   influenced by interaction frequency.'")
    print("="*60)
