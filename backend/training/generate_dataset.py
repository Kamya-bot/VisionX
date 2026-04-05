"""
Synthetic Dataset Generator for VisionX
Generates realistic user behavioral data for ML model training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings


class UserBehaviorDataGenerator:
    """Generate synthetic user behavioral data"""
    
    def __init__(self, n_samples=10000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_user_profiles(self):
        """Generate different user behavior profiles"""
        
        # Define 4 user segments with different behaviors
        segments = ['casual', 'analytical', 'high_intent', 'power_user']
        segment_distribution = [0.3, 0.25, 0.25, 0.2]  # Distribution
        
        user_segments = np.random.choice(
            segments,
            size=self.n_samples,
            p=segment_distribution
        )
        
        data = []
        
        for i, segment in enumerate(user_segments):
            user_id = f"user_{i+1:05d}"
            
            if segment == 'casual':
                # Casual browsers - low engagement
                session_time = np.random.randint(30, 300)  # 0.5-5 min
                clicks = np.random.randint(1, 10)
                scroll_depth = np.random.uniform(0.1, 0.4)
                comparison_count = np.random.randint(0, 3)
                product_views = np.random.randint(1, 5)
                decision_time = np.random.randint(300, 900)  # 5-15 min
                price_sensitivity = np.random.uniform(0.6, 0.9)
                feature_interest = np.random.uniform(0.2, 0.5)
                engagement = np.random.uniform(0.1, 0.4)
                purchase_intent = np.random.uniform(0.1, 0.3)
                categories = np.random.randint(1, 3)
                previous_decisions = np.random.randint(0, 2)
                device = np.random.choice(['mobile', 'tablet', 'desktop'], p=[0.6, 0.2, 0.2])
                
            elif segment == 'analytical':
                # Analytical researchers - high research, moderate speed
                session_time = np.random.randint(600, 1800)  # 10-30 min
                clicks = np.random.randint(15, 40)
                scroll_depth = np.random.uniform(0.7, 0.95)
                comparison_count = np.random.randint(3, 8)
                product_views = np.random.randint(8, 20)
                decision_time = np.random.randint(1200, 2400)  # 20-40 min
                price_sensitivity = np.random.uniform(0.5, 0.7)
                feature_interest = np.random.uniform(0.7, 0.95)
                engagement = np.random.uniform(0.7, 0.9)
                purchase_intent = np.random.uniform(0.5, 0.7)
                categories = np.random.randint(3, 6)
                previous_decisions = np.random.randint(2, 10)
                device = np.random.choice(['mobile', 'tablet', 'desktop'], p=[0.2, 0.2, 0.6])
                
            elif segment == 'high_intent':
                # High intent buyers - focused, ready to buy
                session_time = np.random.randint(300, 900)  # 5-15 min
                clicks = np.random.randint(10, 25)
                scroll_depth = np.random.uniform(0.5, 0.8)
                comparison_count = np.random.randint(2, 5)
                product_views = np.random.randint(5, 12)
                decision_time = np.random.randint(300, 900)  # 5-15 min
                price_sensitivity = np.random.uniform(0.3, 0.6)
                feature_interest = np.random.uniform(0.6, 0.85)
                engagement = np.random.uniform(0.7, 0.95)
                purchase_intent = np.random.uniform(0.7, 0.95)
                categories = np.random.randint(2, 4)
                previous_decisions = np.random.randint(3, 15)
                device = np.random.choice(['mobile', 'tablet', 'desktop'], p=[0.4, 0.2, 0.4])
                
            else:  # power_user
                # Power users - efficient, experienced
                session_time = np.random.randint(180, 600)  # 3-10 min
                clicks = np.random.randint(8, 20)
                scroll_depth = np.random.uniform(0.4, 0.7)
                comparison_count = np.random.randint(1, 4)
                product_views = np.random.randint(3, 8)
                decision_time = np.random.randint(120, 600)  # 2-10 min
                price_sensitivity = np.random.uniform(0.4, 0.7)
                feature_interest = np.random.uniform(0.5, 0.8)
                engagement = np.random.uniform(0.6, 0.85)
                purchase_intent = np.random.uniform(0.6, 0.9)
                categories = np.random.randint(2, 5)
                previous_decisions = np.random.randint(10, 50)
                device = np.random.choice(['mobile', 'tablet', 'desktop'], p=[0.3, 0.1, 0.6])
            
            # Add some noise to make data more realistic
            data.append({
                'user_id': user_id,
                'session_time': int(session_time * np.random.uniform(0.9, 1.1)),
                'clicks': int(clicks * np.random.uniform(0.8, 1.2)),
                'scroll_depth': np.clip(scroll_depth + np.random.normal(0, 0.05), 0, 1),
                'categories_viewed': categories,
                'comparison_count': int(comparison_count),
                'product_views': int(product_views),
                'decision_time': int(decision_time * np.random.uniform(0.9, 1.1)),
                'price_sensitivity': np.clip(price_sensitivity + np.random.normal(0, 0.05), 0, 1),
                'feature_interest_score': np.clip(feature_interest + np.random.normal(0, 0.05), 0, 1),
                'device_type': device,
                'previous_decisions': int(previous_decisions),
                'engagement_score': np.clip(engagement + np.random.normal(0, 0.05), 0, 1),
                'purchase_intent_score': np.clip(purchase_intent + np.random.normal(0, 0.05), 0, 1),
                'true_segment': segment
            })
        
        return pd.DataFrame(data)
    
    def add_derived_features(self, df):
        """Add derived features for better ML performance"""
        
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
        
        # Intent signal
        df['intent_signal'] = (df['purchase_intent_score'] * 0.4 + 
                               df['engagement_score'] * 0.3 + 
                               df['decision_efficiency'] * 0.3)
        
        return df
    
    def generate_dataset(self):
        """Generate complete dataset"""
        print(f"🔄 Generating {self.n_samples} user behavior samples...")
        
        # Generate base data
        df = self.generate_user_profiles()
        
        # Add derived features
        df = self.add_derived_features(df)
        
        print(f"✅ Generated dataset with {len(df)} samples and {len(df.columns)} features")
        
        return df
    
    def save_dataset(self, df, filename='user_behavior_dataset.csv'):
        """Save dataset to file"""
        filepath = os.path.join(settings.RAW_DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"💾 Dataset saved to: {filepath}")
        
        # Print summary statistics
        print("\n📊 Dataset Summary:")
        print(f"  - Total samples: {len(df)}")
        print(f"  - Features: {len(df.columns)}")
        print(f"  - User segments: {df['true_segment'].value_counts().to_dict()}")
        print(f"  - Device types: {df['device_type'].value_counts().to_dict()}")
        
        return filepath


def main():
    """Main execution function"""
    
    # Create directories
    from app.config import create_directories
    create_directories()
    
    # Generate dataset
    generator = UserBehaviorDataGenerator(n_samples=10000, random_state=42)
    df = generator.generate_dataset()
    
    # Save dataset
    filepath = generator.save_dataset(df)
    
    # Display sample
    print("\n📋 Sample Data (first 5 rows):")
    print(df.head())
    
    print("\n📈 Feature Statistics:")
    print(df.describe())
    
    print("\n✅ Dataset generation complete!")
    print(f"📁 File location: {filepath}")
    

if __name__ == "__main__":
    main()
