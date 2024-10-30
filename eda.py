import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def perform_eda(df):
    """
    Perform Exploratory Data Analysis on the fitness app dataset
    """
    
    # 1. Basic Dataset Information
    print("\n=== Basic Dataset Information ===")
    print("\nDataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)
    
    # 2. Summary Statistics
    print("\n=== Summary Statistics ===")
    print(df.describe())
    
   
    # 4. Visualizations
    
    # 4.1 Distribution of Numerical Variables
    numerical_vars = ['Age', 'App Sessions', 'Distance Travelled (km)', 'Calories Burned']
    
    plt.figure(figsize=(15, 5))
    for i, var in enumerate(numerical_vars, 1):
        plt.subplot(1, 4, i)
        plt.hist(df[var], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'{var} Distribution')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 4.2 Box Plots for Categorical Variables
    categorical_vars = ['Gender', 'Activity Level', 'Location']
    
    plt.figure(figsize=(15, 5))
    for i, var in enumerate(categorical_vars, 1):
        plt.subplot(1, 3, i)
        df.boxplot(column='App Sessions', by=var)
        plt.title(f'App Sessions by {var}')
        plt.xticks(rotation=45)
        plt.suptitle('')  # This removes the automatic suptitle
    plt.tight_layout()
    plt.show()
    
    # 4.3 Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numerical_df = df[numerical_vars]
    correlation_matrix = numerical_df.corr()
    
    plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(numerical_vars)), numerical_vars, rotation=45)
    plt.yticks(range(len(numerical_vars)), numerical_vars)
    
    # Add correlation values
    for i in range(len(numerical_vars)):
        for j in range(len(numerical_vars)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center')
    
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    
    # 4.4 Scatter Plots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(df['Age'], df['App Sessions'], alpha=0.5)
    plt.title('Age vs App Sessions')
    plt.xlabel('Age')
    plt.ylabel('App Sessions')
    
    plt.subplot(1, 3, 2)
    plt.scatter(df['Distance Travelled (km)'], df['App Sessions'], alpha=0.5)
    plt.title('Distance vs App Sessions')
    plt.xlabel('Distance Travelled (km)')
    plt.ylabel('App Sessions')
    
    plt.subplot(1, 3, 3)
    plt.scatter(df['Calories Burned'], df['App Sessions'], alpha=0.5)
    plt.title('Calories vs App Sessions')
    plt.xlabel('Calories Burned')
    plt.ylabel('App Sessions')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Statistical Analysis
    print("\n=== Statistical Analysis ===")
    
    # 5.1 User Demographics
    print("\nUser Demographics:")
    print("\nGender Distribution:")
    print(df['Gender'].value_counts(normalize=True).round(3))
    
    print("\nActivity Level Distribution:")
    print(df['Activity Level'].value_counts(normalize=True).round(3))
    
    print("\nLocation Distribution:")
    print(df['Location'].value_counts(normalize=True).round(3))
    
    # 5.2 App Usage Analysis
    print("\nApp Usage Analysis:")
    print("\nMean App Sessions by Activity Level:")
    print(df.groupby('Activity Level')['App Sessions'].mean().round(2))
    
    print("\nMean App Sessions by Location:")
    print(df.groupby('Location')['App Sessions'].mean().round(2))
    
    print("\nMean App Sessions by Gender:")
    print(df.groupby('Gender')['App Sessions'].mean().round(2))
    
    # 5.3 Correlation Analysis
    print("\nCorrelation with App Sessions:")
    correlations = df[numerical_vars].corr()['App Sessions'].sort_values(ascending=False)
    print(correlations)

# Load and analyze data
if __name__ == "__main__":
    df = pd.read_csv('dataset.csv')
    perform_eda(df)