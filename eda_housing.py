# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
# Since you're downloading from Kaggle, save the file as 'housing.csv' in your working directory
df = pd.read_csv('housing.csv')

# Display basic information about the dataset
print("="*50)
print("DATASET OVERVIEW")
print("="*50)
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nDataset info:")
print(df.info())
print(f"\nStatistical summary:")
print(df.describe())

# Check for missing values
print("\n" + "="*50)
print("MISSING VALUES ANALYSIS")
print("="*50)
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})
print(missing_df[missing_df['Missing Values'] > 0])

# Handle missing values (for total_bedrooms)
# Option 1: Fill with median
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

# Verify no missing values remain
print(f"\nRemaining missing values: {df.isnull().sum().sum()}")

# ============================================================================
# 1. DISTRIBUTION PLOTS
# ============================================================================
print("\n" + "="*50)
print("1. DISTRIBUTION ANALYSIS")
print("="*50)

# Set up the plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create distribution plots for numerical features
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    if idx < len(axes):
        # Histogram with KDE
        axes[idx].hist(df[col], bins=50, edgecolor='black', alpha=0.7, density=True)
        axes[idx].axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[col].mean():.2f}')
        axes[idx].axvline(df[col].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[col].median():.2f}')
        axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Density')
        axes[idx].legend()
        
        # Add skewness information
        skewness = df[col].skew()
        axes[idx].text(0.05, 0.95, f'Skewness: {skewness:.2f}', 
                      transform=axes[idx].transAxes, fontsize=9,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Distribution Plots with Mean and Median', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Box plots for outlier visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    if idx < len(axes):
        axes[idx].boxplot(df[col], vert=True)
        axes[idx].set_title(f'Boxplot of {col}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(col)
        
        # Add outlier count
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)].shape[0]
        axes[idx].text(0.05, 0.95, f'Outliers: {outliers}', 
                      transform=axes[idx].transAxes, fontsize=9,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

plt.suptitle('Boxplots for Outlier Detection', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ============================================================================
# 2. CORRELATION HEATMAP
# ============================================================================
print("\n" + "="*50)
print("2. CORRELATION ANALYSIS")
print("="*50)

# Calculate correlation matrix
correlation_matrix = df[numerical_cols].corr()

# Display correlation with target variable (median_house_value)
print("\nCorrelation with Median House Value:")
correlation_with_target = correlation_matrix['median_house_value'].sort_values(ascending=False)
print(correlation_with_target)

# Create heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# ============================================================================
# 3. SCATTER PLOTS
# ============================================================================
print("\n" + "="*50)
print("3. SCATTER PLOT ANALYSIS")
print("="*50)

# Create scatter plots for key relationships
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

# Scatter plot 1: Median Income vs House Value
axes[0].scatter(df['median_income'], df['median_house_value'], alpha=0.5, s=10)
axes[0].set_xlabel('Median Income (in tens of thousands of USD)', fontsize=11)
axes[0].set_ylabel('Median House Value (USD)', fontsize=11)
axes[0].set_title('Median Income vs House Value', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Scatter plot 2: Housing Median Age vs House Value
axes[1].scatter(df['housing_median_age'], df['median_house_value'], alpha=0.5, s=10, c='green')
axes[1].set_xlabel('Housing Median Age', fontsize=11)
axes[1].set_ylabel('Median House Value (USD)', fontsize=11)
axes[1].set_title('Housing Age vs House Value', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Scatter plot 3: Total Rooms vs House Value
axes[2].scatter(df['total_rooms'], df['median_house_value'], alpha=0.5, s=10, c='red')
axes[2].set_xlabel('Total Rooms', fontsize=11)
axes[2].set_ylabel('Median House Value (USD)', fontsize=11)
axes[2].set_title('Total Rooms vs House Value', fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3)

# Scatter plot 4: Population vs House Value
axes[3].scatter(df['population'], df['median_house_value'], alpha=0.5, s=10, c='purple')
axes[3].set_xlabel('Population', fontsize=11)
axes[3].set_ylabel('Median House Value (USD)', fontsize=11)
axes[3].set_title('Population vs House Value', fontsize=13, fontweight='bold')
axes[3].grid(True, alpha=0.3)

plt.suptitle('Key Scatter Plots with Median House Value', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Geographical scatter plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['longitude'], df['latitude'], 
                     c=df['median_house_value'], 
                     cmap='viridis', 
                     alpha=0.6, 
                     s=df['population']/100,  # Size represents population
                     edgecolors='black', 
                     linewidth=0.5)
plt.colorbar(scatter, label='Median House Value (USD)')
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.title('Geographical Distribution of House Prices\n(Point size = Population)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# 4. OUTLIER IDENTIFICATION
# ============================================================================
print("\n" + "="*50)
print("4. OUTLIER IDENTIFICATION")
print("="*50)

# Function to detect outliers using IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Detect and report outliers for each numerical column
outlier_summary = []
for col in numerical_cols:
    outliers, lower_bound, upper_bound = detect_outliers_iqr(df, col)
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(df)) * 100
    
    outlier_summary.append({
        'Feature': col,
        'Outlier Count': outlier_count,
        'Percentage': f'{outlier_percentage:.2f}%',
        'Lower Bound': f'{lower_bound:.2f}',
        'Upper Bound': f'{upper_bound:.2f}'
    })
    
    print(f"\n{col.upper()}:")
    print(f"  - Outliers detected: {outlier_count} ({outlier_percentage:.2f}%)")
    print(f"  - IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

# Create outlier summary dataframe
outlier_df = pd.DataFrame(outlier_summary)
print("\n" + "="*50)
print("OUTLIER SUMMARY TABLE")
print("="*50)
print(outlier_df.to_string(index=False))

# Visualize outliers using boxplots with more detail
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    if idx < len(axes):
        # Create boxplot with outlier points
        bp = axes[idx].boxplot(df[col], vert=True, patch_artist=True)
        axes[idx].set_title(f'Outliers in {col}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(col)
        
        # Color the box
        bp['boxes'][0].set_facecolor('lightblue')
        
        # Add statistical info
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        stats_text = f'Q1: {Q1:.2f}\nQ3: {Q3:.2f}\nIQR: {IQR:.2f}'
        axes[idx].text(0.05, 0.95, stats_text, transform=axes[idx].transAxes, 
                      fontsize=8, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Detailed Outlier Analysis with Boxplots', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# ============================================================================
# 5. ADDITIONAL ANALYSIS - Ocean Proximity
# ============================================================================
print("\n" + "="*50)
print("5. OCEAN PROXIMITY ANALYSIS")
print("="*50)

# Analyze median house value by ocean proximity
print("\nMedian House Value by Ocean Proximity:")
ocean_proximity_stats = df.groupby('ocean_proximity')['median_house_value'].agg(['mean', 'median', 'count'])
print(ocean_proximity_stats.round(2))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Boxplot
df.boxplot(column='median_house_value', by='ocean_proximity', ax=axes[0])
axes[0].set_title('House Value Distribution by Ocean Proximity', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Ocean Proximity')
axes[0].set_ylabel('Median House Value (USD)')
axes[0].tick_params(axis='x', rotation=45)

# Count plot
ocean_counts = df['ocean_proximity'].value_counts()
axes[1].bar(ocean_counts.index, ocean_counts.values, color='skyblue', edgecolor='black')
axes[1].set_title('Number of Properties by Ocean Proximity', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Ocean Proximity')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)

# Add value labels on bars
for i, v in enumerate(ocean_counts.values):
    axes[1].text(i, v + 100, str(v), ha='center', fontweight='bold')

plt.suptitle('Ocean Proximity Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("EDA COMPLETED SUCCESSFULLY!")
print("="*50)