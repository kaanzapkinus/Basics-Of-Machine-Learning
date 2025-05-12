import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

file = os.path.join(os.path.dirname(__file__), 'LifeExpectancy.csv')
df = pd.read_csv(file, header=0)

#y = x + b

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
regression = linear_model.LinearRegression(fit_intercept=True)

# Print missing values information
print("Missing values in dataset:")
print(df[['GDP', 'Total expenditure', 'Alcohol', ' HIV/AIDS', 'Life expectancy']].isna().sum())

# Handle missing values - creating a copy to avoid warnings
df_processed = df.copy()

# Create imputer to fill missing values with mean
imputer = SimpleImputer(strategy='mean')

# Get feature names for later use
feature_names = ['GDP', 'Total expenditure', 'Alcohol', ' HIV/AIDS']

# Apply imputer to features
X_data = imputer.fit_transform(df_processed[feature_names])

# Apply imputer to target (if any missing values)
y_data = df_processed['Life expectancy'].values
y_data = y_data.reshape(-1, 1)  # Reshape for imputer
y_data = SimpleImputer(strategy='mean').fit_transform(y_data).ravel()  # Flatten back to 1D

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_data, 
    y_data,  
    test_size=0.2,  
    random_state=38
)

# Train multiple regression model
model = regression.fit(X_train, y_train)

print("\nMultiple Regression Model Results:")
print("Model coefficients:", model.coef_)
print("Training set score:", model.score(X_train, y_train))
print("Test set score:", model.score(X_test, y_test))

# 1. Data Analysis and Visualization
print("\nNumber of records in training set:", len(X_train))
print("Number of records in test set:", len(X_test))

# Life expectancy histogram
plt.figure(figsize=(10, 5))
plt.hist(df['Life expectancy'], bins=30, color='skyblue', edgecolor='black')
plt.title('Life Expectancy Distribution')
plt.xlabel('Life Expectancy')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# Statistical information
print("\nLife Expectancy Statistics:")
print(df['Life expectancy'].describe())

# Top 3 countries with highest life expectancy
top_3_countries = df.nlargest(3, 'Life expectancy')[['Country', 'Life expectancy']]
print("\nTop 3 Countries with Highest Life Expectancy:")
print(top_3_countries)

# 2. Simple Regression Models
features = ['GDP', 'Total expenditure', 'Alcohol', ' HIV/AIDS']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, feature in enumerate(features):
    # Model creation
    X_simple = df_processed[[feature]]
    X_simple = imputer.fit_transform(X_simple)  # Handle missing values
    y_simple = df_processed['Life expectancy'].values
    y_simple = y_simple.reshape(-1, 1)
    y_simple = imputer.fit_transform(y_simple).ravel()
    
    # Train-test split
    X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
        X_simple, y_simple, test_size=0.2, random_state=38
    )
    
    # Model training
    model_simple = linear_model.LinearRegression()
    model_simple.fit(X_train_simple, y_train_simple)
    
    # Plotting
    axes[i].scatter(X_train_simple, y_train_simple, color='blue', alpha=0.5, label='Training Data')
    axes[i].plot(X_train_simple, model_simple.predict(X_train_simple), color='red', label='Regression Line')
    
    # Regression equation
    equation = f'y = {model_simple.coef_[0]:.2f}x + {model_simple.intercept_:.2f}'
    axes[i].set_title(f'{feature} vs Life Expectancy\n{equation}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Life Expectancy')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    
    # Model scores
    print(f"\n{feature} Simple Regression Model Scores:")
    print(f"Training set score: {model_simple.score(X_train_simple, y_train_simple):.3f}")
    print(f"Test set score: {model_simple.score(X_test_simple, y_test_simple):.3f}")

plt.tight_layout()
plt.show()

# 3. Prediction and Error Analysis
y_pred = model.predict(X_test)
errors = y_test - y_pred

print("\nMultiple Regression Error Analysis:")
print(f"Mean Error: {np.mean(errors):.2f}")
print(f"Error Standard Deviation: {np.std(errors):.2f}")
print(f"Mean Absolute Error: {np.mean(np.abs(errors)):.2f}")

# Error distribution plot
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, color='lightgreen', edgecolor='black')
plt.title('Distribution of Prediction Errors')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# 4. Correlation Analysis
correlation_matrix = df[['GDP', 'Total expenditure', 'Alcohol', ' HIV/AIDS', 'Life expectancy']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Sorting by correlation values
correlations = correlation_matrix['Life expectancy'].sort_values(ascending=False)
print("\nCorrelations with Life Expectancy:")
print(correlations)

# Correlation matrix visualization
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Matrix Between Variables')

# Adding correlation values to matrix
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                ha='center', va='center')

plt.tight_layout()
plt.show()

# Final Report
print("\nFinal Report:")
print("="*50)
print("1. Data Analysis Summary:")
print(f"- Total number of records: {len(df)}")
print(f"- Training set size: {len(X_train)}")
print(f"- Test set size: {len(X_test)}")
print(f"- Features used: {feature_names}")

print("\n2. Model Performance Summary:")
print("Simple Regression Models:")
for feature in feature_names:
    X_simple = df_processed[[feature]]
    X_simple = imputer.fit_transform(X_simple)
    y_simple = df_processed['Life expectancy'].values
    y_simple = y_simple.reshape(-1, 1)
    y_simple = imputer.fit_transform(y_simple).ravel()
    
    X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
        X_simple, y_simple, test_size=0.2, random_state=38
    )
    
    model_simple = linear_model.LinearRegression()
    model_simple.fit(X_train_simple, y_train_simple)
    print(f"- {feature}:")
    print(f"  * Training score: {model_simple.score(X_train_simple, y_train_simple):.3f}")
    print(f"  * Test score: {model_simple.score(X_test_simple, y_test_simple):.3f}")

print("\nMultiple Linear Regression Model:")
print(f"  * Training score: {model.score(X_train, y_train):.3f}")
print(f"  * Test score: {model.score(X_test, y_test):.3f}")

print("\n3. Feature Importance:")
for feature, coef in zip(feature_names, model.coef_):
    print(f"- {feature}: {coef:.3f}")

print("\n4. Error Analysis Summary:")
print(f"- Mean Error: {np.mean(errors):.2f}")
print(f"- Error Standard Deviation: {np.std(errors):.2f}")
print(f"- Mean Absolute Error: {np.mean(np.abs(errors)):.2f}")

print("\n5. Conclusions:")
print("- The multiple regression model (using 4 features) shows improved performance compared to simple regression models")
print(f"- The most important feature is {feature_names[np.argmax(np.abs(model.coef_))]} with coefficient {model.coef_[np.argmax(np.abs(model.coef_))]:.3f}")
print(f"- The model explains {model.score(X_test, y_test)*100:.1f}% of the variance in life expectancy in the test set")