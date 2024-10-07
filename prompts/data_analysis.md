# Data Analysis Instructions

1. Load the dataset from 'personalized_loan_recommendation_dataset.csv'.
2. Perform Exploratory Data Analysis (EDA):
   - Check for missing values and handle them appropriately
   - Analyze the distribution of numerical features
   - Examine the correlation between features
   - Visualize key relationships using plots (e.g., scatter plots, histograms)
3. Preprocess the data:
   - Encode categorical variables (e.g., one-hot encoding for 'employment_status', 'home_ownership', etc.)
   - Normalize numerical features
   - Split the data into features (X) and target variable (y)
4. Perform feature selection:
   - Use techniques like correlation analysis, mutual information, or feature importance from tree-based models
   - Select the most relevant features for predicting loan approval
5. Split the data into training and testing sets (e.g., 80% train, 20% test).
6. Save the preprocessed data and selected features for use in model building.

Note: Ensure all data preprocessing steps are documented for reproducibility.