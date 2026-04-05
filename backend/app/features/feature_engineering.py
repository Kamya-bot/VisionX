"""
Feature Engineering Pipeline for VisionX ML Backend
Transforms raw behavioral data into ML-ready features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings


class FeatureEngineer:
    """Feature engineering and preprocessing pipeline"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.categorical_columns = ['device_type']
        self.numerical_columns = [
            'session_time', 'clicks', 'scroll_depth', 'categories_viewed',
            'comparison_count', 'product_views', 'decision_time',
            'price_sensitivity', 'feature_interest_score', 'previous_decisions',
            'engagement_score', 'purchase_intent_score', 'engagement_ratio',
            'decision_efficiency', 'interaction_score', 'behavior_intensity',
            'research_depth', 'intent_signal'
        ]
        
    def create_derived_features(self, df):
        """Create derived features from raw data"""
        
        print("🔧 Creating derived features...")
        
        # Engagement ratio
        df['engagement_ratio'] = df['clicks'] / (df['session_time'] + 1)
        
        # Decision efficiency
        df['decision_efficiency'] = df['product_views'] / (df['decision_time'] + 1)
        
        # Interaction score
        df['interaction_score'] = df['clicks'] * df['scroll_depth']
        
        # Behavior intensity
        df['behavior_intensity'] = df['comparison_count'] + df['product_views']
        
        # Research depth
        df['research_depth'] = df['categories_viewed'] * df['scroll_depth']
        
        # Intent signal (weighted combination)
        df['intent_signal'] = (
            df['purchase_intent_score'] * 0.4 + 
            df['engagement_score'] * 0.3 + 
            df['decision_efficiency'] * 0.3
        )
        
        # Experience level
        df['experience_level'] = np.log1p(df['previous_decisions'])
        
        # Session efficiency
        df['session_efficiency'] = df['product_views'] / (df['session_time'] + 1)
        
        # Decision speed category
        df['decision_speed'] = pd.cut(
            df['decision_time'],
            bins=[0, 300, 900, 1800, np.inf],
            labels=['fast', 'moderate', 'slow', 'very_slow']
        )
        
        print(f"✅ Created {len(df.columns)} total features")
        
        return df
    
    def handle_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        
        print("🔤 Encoding categorical features...")
        
        # One-hot encode device_type
        if 'device_type' in df.columns:
            device_dummies = pd.get_dummies(df['device_type'], prefix='device')
            df = pd.concat([df, device_dummies], axis=1)
        
        # Encode decision_speed if present
        if 'decision_speed' in df.columns:
            speed_dummies = pd.get_dummies(df['decision_speed'], prefix='speed')
            df = pd.concat([df, speed_dummies], axis=1)
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        
        print("🔍 Checking for missing values...")
        
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"⚠️  Found {missing.sum()} missing values")
            
            # Fill numerical with median
            for col in self.numerical_columns:
                if col in df.columns and df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
            
            # Fill categorical with mode
            for col in self.categorical_columns:
                if col in df.columns and df[col].isnull().any():
                    df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            print("✅ No missing values found")
        
        return df
    
    def handle_outliers(self, df, columns=None, n_std=3):
        """Remove or cap outliers"""
        
        print("📊 Handling outliers...")
        
        if columns is None:
            columns = self.numerical_columns
        
        outliers_removed = 0
        
        for col in columns:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - n_std * std
                upper_bound = mean + n_std * std
                
                # Cap outliers instead of removing
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    outliers_removed += outliers
        
        if outliers_removed > 0:
            print(f"✅ Capped {outliers_removed} outliers")
        else:
            print("✅ No significant outliers found")
        
        return df
    
    def normalize_features(self, df, fit=True):
        """Normalize numerical features"""
        
        print("📏 Normalizing numerical features...")
        
        # Select only numerical columns that exist in df
        numerical_cols = [col for col in self.numerical_columns if col in df.columns]
        
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            print("✅ Fitted and transformed features")
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            print("✅ Transformed features using existing scaler")
        
        return df
    
    def get_feature_matrix(self, df):
        """Extract feature matrix for ML"""
        
        # Define feature columns (excluding metadata and target)
        exclude_cols = ['user_id', 'true_segment', 'device_type', 'decision_speed']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols].values
        
        print(f"✅ Feature matrix shape: {X.shape}")
        
        return X, feature_cols
    
    def process_pipeline(self, df, fit=True):
        """Complete feature engineering pipeline"""
        
        print("\n" + "="*60)
        print("🚀 FEATURE ENGINEERING PIPELINE")
        print("="*60 + "\n")
        
        # 1. Handle missing values
        df = self.handle_missing_values(df)
        
        # 2. Create derived features
        df = self.create_derived_features(df)
        
        # 3. Handle outliers
        df = self.handle_outliers(df)
        
        # 4. Encode categorical features
        df = self.handle_categorical_features(df, fit=fit)
        
        # 5. Normalize numerical features
        df = self.normalize_features(df, fit=fit)
        
        # 6. Get feature matrix
        X, feature_cols = self.get_feature_matrix(df)
        
        print("\n" + "="*60)
        print("✅ FEATURE ENGINEERING COMPLETE")
        print("="*60)
        
        return df, X, feature_cols
    
    def save_preprocessors(self):
        """Save fitted preprocessors"""
        
        # Save scaler
        scaler_path = settings.SCALER_PATH
        joblib.dump(self.scaler, scaler_path)
        print(f"💾 Scaler saved to: {scaler_path}")
        
        # Save feature columns
        feature_cols_path = os.path.join(settings.MODEL_DIR, 'feature_columns.pkl')
        joblib.dump(self.feature_columns, feature_cols_path)
        print(f"💾 Feature columns saved to: {feature_cols_path}")
    
    def load_preprocessors(self):
        """Load fitted preprocessors"""
        
        # Load scaler
        scaler_path = settings.SCALER_PATH
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Scaler loaded from: {scaler_path}")
        
        # Load feature columns
        feature_cols_path = os.path.join(settings.MODEL_DIR, 'feature_columns.pkl')
        if os.path.exists(feature_cols_path):
            self.feature_columns = joblib.load(feature_cols_path)
            print(f"✅ Feature columns loaded from: {feature_cols_path}")


def main():
    """Test feature engineering pipeline"""
    
    # Load raw dataset
    raw_data_path = os.path.join(settings.RAW_DATA_DIR, 'user_behavior_dataset.csv')
    
    if not os.path.exists(raw_data_path):
        print(f"❌ Dataset not found at: {raw_data_path}")
        print("Please run generate_dataset.py first")
        return
    
    print(f"📂 Loading dataset from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    print(f"✅ Loaded {len(df)} samples")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Process data
    df_processed, X, feature_cols = engineer.process_pipeline(df, fit=True)
    
    # Save processed data
    processed_path = os.path.join(settings.PROCESSED_DATA_DIR, 'features.csv')
    df_processed.to_csv(processed_path, index=False)
    print(f"\n💾 Processed data saved to: {processed_path}")
    
    # Save preprocessors
    engineer.save_preprocessors()
    
    # Display results
    print("\n📊 Processing Summary:")
    print(f"  - Original shape: {df.shape}")
    print(f"  - Processed shape: {df_processed.shape}")
    print(f"  - Feature matrix shape: {X.shape}")
    print(f"  - Number of features: {len(feature_cols)}")
    
    print("\n📋 Feature List:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    print("\n✅ Feature engineering complete!")


if __name__ == "__main__":
    main()
