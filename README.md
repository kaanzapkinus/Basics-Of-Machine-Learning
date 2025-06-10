# Basics of Machine Learning - Final Project

This repository contains the final project for the "Basics of Machine Learning" course. The project focuses on predicting **rental prices** for apartments in **Antalya, Turkey**, using machine learning models. The dataset used includes apartment listings from the Muratpaşa district.

### **Project Overview**

The goal of this project was to develop a machine learning model that predicts rental prices based on various apartment features. These features include the number of rooms, building age, usable area (m²), elevator availability, and more. The project involved:

- Data **cleaning** and **preprocessing**.
- Feature engineering, including the creation of new features.
- Development and comparison of machine learning models.
- Model **evaluation** and **error analysis**.

### **Key Steps in the Project**

1. **Data Import and Preprocessing:**
   - Imported dataset from Kaggle containing rental listings.
   - Cleaned the data by handling missing values and encoding categorical features like room count and building age.
   - Removed outliers to improve model accuracy.

2. **Machine Learning Models Used:**
   - **Linear Regression:** A simple approach to model the relationship between features and rental price.
   - **Random Forest Regressor:** A more complex, tree-based ensemble model that captures non-linear relationships.

3. **Model Evaluation:**
   - The performance of both models was evaluated using metrics such as **R²**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)**.
   - The Random Forest model outperformed Linear Regression with a test **R² score of 0.490**.

4. **Feature Importance:**
   - The most important features influencing the rental prices were **net area** and **monthly fee**, according to the Random Forest model’s feature importance analysis.

### **What I Learned**

- **Data Preprocessing**: Gained hands-on experience in handling real-world data, cleaning it, and engineering new features for better model performance.
- **Model Development**: Learned how to apply different machine learning algorithms and assess their performance.
- **Feature Engineering**: Realized the importance of selecting the right features for improving model accuracy.
- **Error Analysis**: Understood the significance of analyzing prediction errors to improve model performance in future iterations.

### **Improvements for Future Work**

- Add geolocation and neighborhood-level data to improve predictions.
- Incorporate additional features such as **heating type**, **parking availability**, and **floor level**.
- Explore advanced models like **Gradient Boosting** and **XGBoost** for better accuracy.
- Gather a larger dataset for better generalization and model robustness.
