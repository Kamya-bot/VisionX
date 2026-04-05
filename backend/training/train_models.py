"""
VisionX ML Model Training Pipeline
Trains clustering, prediction, and recommendation models
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings, create_directories
from app.features.feature_engineering import FeatureEngineer


class VisionXModelTrainer:
    """Train all ML models for VisionX"""
    
    def __init__(self):
        self.clustering_model = None
        self.prediction_model = None
        self.feature_engineer = FeatureEngineer()
        self.results = {}
        
    def load_data(self):
        """Load and prepare data"""
        
        print("\n" + "="*70)
        print("📂 LOADING DATA")
        print("="*70)
        
        raw_data_path = os.path.join(settings.RAW_DATA_DIR, 'user_behavior_dataset.csv')
        
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(
                f"Dataset not found at: {raw_data_path}\n"
                "Please run: python training/generate_dataset.py"
            )
        
        df = pd.read_csv(raw_data_path)
        print(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
        
        return df
    
    def train_clustering_model(self, X, df):
        """Train K-Means clustering model"""
        
        print("\n" + "="*70)
        print("🎯 TRAINING CLUSTERING MODEL (K-Means)")
        print("="*70)
        
        # Find optimal K using elbow method
        print("\n🔍 Finding optimal number of clusters...")
        inertias = []
        silhouette_scores = []
        K_range = range(2, 8)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=settings.RANDOM_SEED, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        
        # Plot elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(K_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (K)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True)
        
        ax2.plot(K_range, silhouette_scores, 'go-')
        ax2.set_xlabel('Number of Clusters (K)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(settings.MODEL_DIR, 'elbow_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"📊 Elbow analysis saved to: {settings.MODEL_DIR}/elbow_analysis.png")
        
        # Train final model with optimal K
        optimal_k = settings.N_CLUSTERS
        print(f"\n🎯 Training K-Means with K={optimal_k}...")
        
        self.clustering_model = KMeans(
            n_clusters=optimal_k,
            random_state=settings.RANDOM_SEED,
            max_iter=settings.MAX_ITERATIONS,
            n_init=10
        )
        
        cluster_labels = self.clustering_model.fit_predict(X)
        
        # Evaluate clustering
        silhouette = silhouette_score(X, cluster_labels)
        davies_bouldin = davies_bouldin_score(X, cluster_labels)
        
        print(f"\n📊 Clustering Metrics:")
        print(f"  - Silhouette Score: {silhouette:.4f}")
        print(f"  - Davies-Bouldin Index: {davies_bouldin:.4f}")
        print(f"  - Number of Clusters: {optimal_k}")
        
        # Analyze clusters
        df['cluster'] = cluster_labels
        print(f"\n📈 Cluster Distribution:")
        cluster_counts = df['cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            percentage = (count / len(df)) * 100
            label = settings.CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
            print(f"  - {label}: {count} ({percentage:.1f}%)")
        
        # Visualize clusters (2D projection using first 2 principal components)
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                            cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('User Behavioral Clusters (2D Projection)')
        plt.grid(True, alpha=0.3)
        
        # Add cluster centers
        centers_pca = pca.transform(self.clustering_model.cluster_centers_)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                   c='red', marker='X', s=300, edgecolors='black', linewidths=2,
                   label='Cluster Centers')
        plt.legend()
        
        plt.savefig(os.path.join(settings.MODEL_DIR, 'cluster_visualization.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"📊 Cluster visualization saved to: {settings.MODEL_DIR}/cluster_visualization.png")
        
        # Save results
        self.results['clustering'] = {
            'silhouette_score': float(silhouette),
            'davies_bouldin_index': float(davies_bouldin),
            'n_clusters': optimal_k,
            'cluster_distribution': cluster_counts.to_dict()
        }
        
        # Save model
        model_path = settings.CLUSTERING_MODEL_PATH
        joblib.dump(self.clustering_model, model_path)
        print(f"💾 Clustering model saved to: {model_path}")
        
        return cluster_labels
    
    def train_prediction_model(self, X, y):
        """Train XGBoost prediction model"""
        
        print("\n" + "="*70)
        print("🎯 TRAINING PREDICTION MODEL (XGBoost)")
        print("="*70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_SEED, stratify=y
        )
        
        print(f"\n📊 Data Split:")
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Test samples: {len(X_test)}")
        
        # Train XGBoost
        print(f"\n🚀 Training XGBoost classifier...")
        
        self.prediction_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=settings.RANDOM_SEED,
            eval_metric='mlogloss'
        )
        
        self.prediction_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.prediction_model.predict(X_test)
        y_pred_proba = self.prediction_model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n📊 Prediction Metrics:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=[settings.CLUSTER_LABELS[i] for i in range(settings.N_CLUSTERS)]))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[settings.CLUSTER_LABELS[i] for i in range(settings.N_CLUSTERS)],
                   yticklabels=[settings.CLUSTER_LABELS[i] for i in range(settings.N_CLUSTERS)])
        plt.title('Prediction Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(settings.MODEL_DIR, 'confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {settings.MODEL_DIR}/confusion_matrix.png")
        
        # Feature Importance
        feature_importance = self.prediction_model.feature_importances_
        feature_names = self.feature_engineer.feature_columns
        
        # Get top 15 features
        indices = np.argsort(feature_importance)[-15:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), feature_importance[indices], color='steelblue')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Most Important Features')
        plt.tight_layout()
        plt.savefig(os.path.join(settings.MODEL_DIR, 'feature_importance.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"📊 Feature importance saved to: {settings.MODEL_DIR}/feature_importance.png")
        
        # Save results
        self.results['prediction'] = {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'feature_importance': {feature_names[i]: float(feature_importance[i]) 
                                  for i in range(len(feature_names))}
        }
        
        # Save model
        model_path = settings.PREDICTION_MODEL_PATH
        joblib.dump(self.prediction_model, model_path)
        print(f"💾 Prediction model saved to: {model_path}")
        
        return accuracy, f1
    
    def save_training_results(self):
        """Save training results to JSON"""
        
        results_with_metadata = {
            'training_date': datetime.now().isoformat(),
            'models_trained': ['clustering', 'prediction'],
            'dataset_size': self.results.get('dataset_size', 0),
            'random_seed': settings.RANDOM_SEED,
            'results': self.results
        }
        
        results_path = os.path.join(settings.MODEL_DIR, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"💾 Training results saved to: {results_path}")
    
    def train_all_models(self):
        """Train all models in pipeline"""
        
        print("\n" + "="*70)
        print("🚀 VISIONX ML TRAINING PIPELINE")
        print("="*70)
        
        # Load data
        df = self.load_data()
        self.results['dataset_size'] = len(df)
        
        # Feature engineering
        df_processed, X, feature_cols = self.feature_engineer.process_pipeline(df, fit=True)
        self.feature_engineer.save_preprocessors()
        
        # Train clustering model
        cluster_labels = self.train_clustering_model(X, df_processed)
        
        # Train prediction model (predict cluster membership)
        accuracy, f1 = self.train_prediction_model(X, cluster_labels)
        
        # Save all results
        self.save_training_results()
        
        print("\n" + "="*70)
        print("✅ TRAINING PIPELINE COMPLETE!")
        print("="*70)
        print(f"\n📊 Summary:")
        print(f"  - Clustering Silhouette: {self.results['clustering']['silhouette_score']:.4f}")
        print(f"  - Prediction Accuracy: {self.results['prediction']['accuracy']:.4f}")
        print(f"  - Prediction F1 Score: {self.results['prediction']['f1_score']:.4f}")
        print(f"\n📁 Models saved to: {settings.MODEL_DIR}")


def main():
    """Main training execution"""
    
    # Create directories
    create_directories()
    
    # Initialize trainer
    trainer = VisionXModelTrainer()
    
    # Train all models
    trainer.train_all_models()
    
    print("\n✅ All models trained successfully!")
    print("\n🚀 Next steps:")
    print("  1. Review training results in: training_results.json")
    print("  2. Check visualizations in: trained_models/")
    print("  3. Start the API server: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
