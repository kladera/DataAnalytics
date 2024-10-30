import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
def preprocess_data(df):
    data = df.copy()
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])
    data['Activity Level'] = le.fit_transform(data['Activity Level'])
    data['Location'] = le.fit_transform(data['Location'])
    return data

# Calculate and visualize feature importance
def analyze_feature_importance(df):
    # Prepare features and target
    features = ['Age', 'Gender', 'Activity Level', 'Location', 
               'Distance Travelled (km)', 'Calories Burned']
    X = df[features]
    y = df['App Sessions']
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Calculate importance scores
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    })
    importance = importance.sort_values('Importance', ascending=False)  # Sort descending
    
    # Set up colors
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create vertical bars
    bars = plt.bar(importance['Feature'], importance['Importance'], 
                  color=colors, alpha=0.8)
    
    # Customize the plot
    plt.title('Feature Importance in Predicting App Sessions', fontsize=12, pad=15)
    plt.ylabel('Importance Score')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Rotate x-labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    # Print importance scores
    print("\nFeature Importance Rankings:")
    print("-" * 40)
    for idx, row in importance.iterrows():
        print(f"{row['Feature']:<25} {row['Importance']:.3f}")

# Execute analysis
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('dataset.csv')
    
    # Preprocess and analyze
    processed_df = preprocess_data(df)
    analyze_feature_importance(processed_df)