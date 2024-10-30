import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('dataset.csv')

# Select relevant numerical columns for correlation
columns_of_interest = ['Age', 'App Sessions', 'Distance Travelled (km)', 'Calories Burned']
correlation_matrix = df[columns_of_interest].corr()

# Create a figure with larger size
plt.figure(figsize=(10, 8))

# Create heatmap
sns.heatmap(correlation_matrix, 
            annot=True,  # Show correlation values
            cmap='coolwarm',  # Blue-red colormap
            vmin=-1, vmax=1,  # Set correlation range
            center=0,  # Center the colormap at 0
            fmt='.2f',  # Show 2 decimal places
            square=True)  # Make cells square

# Customize the plot
plt.title('Correlation Heatmap of Fitness App Variables', pad=20)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Print correlation values
print("\nCorrelation Matrix:")
print(correlation_matrix.round(3))

# Print key insights
print("\nKey Correlation Insights:")
print("-------------------------")
# Get strongest correlations (excluding self-correlations)
correlations = []
for i in range(len(columns_of_interest)):
    for j in range(i+1, len(columns_of_interest)):
        correlations.append({
            'variables': f"{columns_of_interest[i]} vs {columns_of_interest[j]}",
            'correlation': correlation_matrix.iloc[i,j]
        })

# Sort by absolute correlation value
correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

# Print top correlations
for corr in correlations:
    print(f"{corr['variables']}: {corr['correlation']:.3f}")