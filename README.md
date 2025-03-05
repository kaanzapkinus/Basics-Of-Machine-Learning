# Basics-Of-Machine-Learning
Basics Of Machine Learning Class Materials

# Linear Regression – Practice

## Part 1

You can use all notes and online resources. Just be sure that you understand what you are doing in case of any questions. When you upload your solution, indicate which tasks you completed.

### **1. Read the Data**
- Load the data from the CSV file named **LifeExpectance** into a pandas dataframe.

### **2. Prepare the Data**
- Split records into **train dataset** and **test dataset**.
- You can use functions like `train_test_split` from `sklearn`, write your own function, or manually select records from the years **2003, 2008, and 2013** as the test set.
- Show basic information about the data:
  - **a.** Number of records in each dataset.
  - **b.** Histogram of life expectancy and statistical information (mean, standard deviation, etc.).
  - **c.** Identify three countries with the highest life expectancy.

### **3. Train Simple Regression Models**
- Use data from the train set to fit three models using **simple regression** based on the following parameters:
  - **a.** GDP
  - **b.** Total expenditure
  - **c.** Alcohol consumption

### **4. Analyze Regression Models**
- Find coefficients (slopes and intercepts) and scores of the regression line for each model.
- Show charts with data points from the training set and regression lines.
- Display the equation of the regression line on each chart.

### **5. Make Predictions & Analyze Errors**
- Use models created in step 3 to predict life expectancy for the **test set**.
- Compute the **average error** for all three models as well as the **standard deviation** of the predictions.

### **6. Summarize Results**
- Prepare a report summarizing all information obtained from the data and charts.
- Write short **conclusions** based on your findings.

---

## Part 2

### **1. Select Optimal Parameters**
- Choose **four parameters** that are best suited for predicting life expectancy.
- Justify your selection in the final report.

### **2. Train a Multilinear Regression Model**
- Fit a **multilinear regression model** using the selected four parameters on the training data.

### **3. Evaluate the Model**
- Print **coefficients** and **score** of the model.
- Predict values for the test set.
- Print statistical information about errors (average, standard deviation, etc.).

### **4. Compare Results**
- Compare results with **Part 1, step 5**.
- Write **conclusions** on how the multilinear regression model performed compared to simple regression models.

---

## Submission
- Ensure your code is well-documented and structured.
- Include a summary of your findings in the final report.
- Indicate which tasks you completed in your submission.

