# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and prepare the dataset
data = load_iris()
X = data.data  # Features (sepal length, sepal width, petal length, petal width)
y = data.target  # Target labels (iris species)

# 2. Initial model training and evaluation
# Split data into training (75%) and test (25%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions on both training and test sets
train_predictions = clf.predict(X_train)
test_predictions = clf.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

# Print initial accuracy results
print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# 3. Analysis with different random states
# Test with different random states
random_states = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
train_accuracies = []
test_accuracies = []

print("Training Data Analysis with Different Random States:")
print("--------------------------------------------------")
print("Random State | Train Accuracy | Test Accuracy")
print("--------------------------------------------------")

for state in random_states:
    # Split data with different random states
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=state)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    # Make predictions
    train_predictions = clf.predict(X_train)
    test_predictions = clf.predict(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    
    # Print results for each random state
    print(f"{state:11d} | {train_accuracy:.4f}      | {test_accuracy:.4f}")

# Calculate and print statistics
mean_train_accuracy = np.mean(train_accuracies)
std_train_accuracy = np.std(train_accuracies)
mean_test_accuracy = np.mean(test_accuracies)
std_test_accuracy = np.std(test_accuracies)

print("\nSummary Statistics:")
print("--------------------------------------------------")
print(f"Mean Train Accuracy: {mean_train_accuracy:.4f}")
print(f"Std Train Accuracy:  {std_train_accuracy:.4f}")
print(f"Mean Test Accuracy:  {mean_test_accuracy:.4f}")
print(f"Std Test Accuracy:   {std_test_accuracy:.4f}")
print("--------------------------------------------------")

# 4. Analysis with different test set sizes
# Test model performance with different test set ratios
test_sizes = [0.1, 0.25, 0.5, 0.75, 0.9]
train_size_accuracies = []
test_size_accuracies = []

for size in test_sizes:
    # Split data with different test set sizes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    # Make predictions and calculate accuracies
    train_predictions = clf.predict(X_train)
    test_predictions = clf.predict(X_test)
    
    train_size_accuracies.append(accuracy_score(y_train, train_predictions))
    test_size_accuracies.append(accuracy_score(y_test, test_predictions))

# 5. Visualize test set size analysis
plt.figure(figsize=(10, 6))
plt.plot(test_sizes, train_size_accuracies, marker='o', label='Train Accuracy')
plt.plot(test_sizes, test_size_accuracies, marker='s', label='Test Accuracy')
plt.xlabel('Test Set Ratio')
plt.ylabel('Accuracy')
plt.title('Model Accuracy vs Test Set Ratio')
plt.grid(True)
plt.legend()
plt.show()

# 6. Visualize the decision tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()