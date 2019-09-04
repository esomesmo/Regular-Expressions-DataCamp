## Feature engineering

# You are tasked to predict whether a new cohort of loan applicants are likely to default on their loans. You have a historical dataset and wish to train a classifier on it. You notice that many features are in string format, which is a problem for your classifiers. You hence decide to encode the string columns numerically using LabelEncoder(). The function has been preloaded for you from the preprocessing submodule of sklearn. The dataset credit is also preloaded, as is a list of all column names whose data types are string, stored in non_numeric_columns.

# Instructions
# 100 XP
# Inspect the first three lines of your data using .head().
# For each column in non_numeric_columns, replace the string values with numeric values using LabelEncoder().
# Confirm your code worked by printing the data types in the credit data frame.

# Inspect the first few lines of your data using head()
credit.head(3)

# Create a label encoder for each column. Encode the values
for column in non_numeric_columns:
    le = LabelEncoder()
    credit[column] = le.fit_transform(credit[column])

# Inspect the data types of the columns of the data frame
print(credit.dtypes)

## Your first pipeline
# Your colleague has used AdaBoostClassifier for the credit scoring dataset. You want to also try out a random forest classifier. In this exercise, you will fit this classifier to the data and compare it to AdaBoostClassifier. Make sure to use train/test data splitting to avoid overfitting. The data is preloaded and transformed so that all features are numeric. The features are available as X and the labels as y. The module RandomForestClassifier has also been preloaded.

# Instructions 3/3
# 30 XP
# Use accuracy_score to assess the performance of your classifier. An empty dictionary called accuracies is pre-loaded in your environment.

## Grid search CV for model complexity
# In the last slide, you saw how most classifiers have one or more hyperparameters that control its complexity. You also learned to tune them using GridSearchCV(). In this exercise, you will perfect this skill. You will experiment with:

# The number of trees, n_estimators, in a RandomForestClassifier.
# The maximum depth, max_depth, of the decision trees used in an AdaBoostClassifier.
# The number of nearest neighbors, n_neighbors, in KNeighborsClassifier.
# Instructions 1/3
# 35 XP
# Define the parameter grid as described in the code comment and create a grid object with a RandomForestClassifier().

# Set a range for n_estimators from 10 to 40 in steps of 10
param_grid = {'n_estimators': range(10, 50, 10)}

# Optimize for a RandomForestClassifier using GridSearchCV
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid.fit(X, y)
grid.best_params_

# Adapt your code to optimise n_estimators for an AdaBoostClassifier().

# Define a grid for n_estimators ranging from 1 to 10
param_grid = {'n_estimators': range(1, 11)}

# Optimize for a AdaBoostClassifier using GridSearchCV
grid = GridSearchCV(AdaBoostClassifier(), param_grid, cv=3)
grid.fit(X, y)
grid.best_params_

# Adapt your code to optimise n_neighbors for an KNeighborsClassifier().

# Define a grid for n_neighbors with values 10, 50 and 100
param_grid = {'n_neighbors': [10,50,100]}

# Optimize for KNeighborsClassifier using GridSearchCV
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3)
grid.fit(X, y)
grid.best_params_

## Categorical encodings
# Your colleague has converted the columns in the credit dataset to numeric values using LabelEncoder(). He left one out: credit_history, which records the credit history of the applicant. You want to create two versions of the dataset. One will use LabelEncoder() and another one-hot encoding, for comparison purposes. The feature matrix is available to you as credit. You have LabelEncoder() preloaded and pandas as pd.

# Instructions
# 100 XP
# Encode credit_history using LabelEncoder().
# Concatenate the result to the original frame.
# Create a new data frame by concatenating the 1-hot encoding dummies to the original frame.
# Confirm that 1-hot encoding produces more columns than label encoding.

# Create numeric encoding for credit_history
credit_history_num = LabelEncoder().fit_transform(credit['credit_history'])

# Create a new feature matrix including the numeric encoding
X_num = pd.concat([X, pd.Series(credit_history_num)], axis=1)

# Create new feature matrix with dummies for credit_history
X_hot = pd.concat([X, pd.get_dummies(credit['credit_history'])], axis=1)

# Compare the number of features of the resulting DataFrames
X_hot.shape[1] > X_num.shape[1]

## Feature transformations
# You are discussing the credit dataset with the bank manager. She suggests that the safest loan applications tend to request mid-range credit amounts. Values that are either too low or too high suggest high risk. This means that a non-linear relationship might exist between this variable and the class. You want to test this hypothesis. You will construct a non-linear transformation of the feature. Then, you will assess which of the two features is better at predicting the class using SelectKBest() and the chi2() metric, both of which have been preloaded.

# The data is available as a pandas DataFrame called credit, with the class contained in the column class. You also have preloaded pandas as pd and numpy as np.

# Instructions
# 100 XP
# Define a function that transforms a numeric vector by considering the absolute difference of each value from the average value of the vector.
# Apply this transformation to the credit_amount column of the dataset and store in new column called diff
# Create a SelectKBest() feature selector to pick one of the two columns, credit_amount and diff using the chi2() metric.
# Inspect the results.

# Function computing absolute difference from column mean
def abs_diff(x):
    return np.abs(x-np.mean(x))

# Apply it to the credit amount and store to new column
credit['diff'] = abs_diff(credit['credit_amount'])

# Create a feature selector with chi2 that picks one feature
sk = SelectKBest(chi2, k=1)

# Use the selector to pick between credit_amount and diff
sk.fit(credit[['credit_amount','diff']], credit['class'])

# Inspect the results
sk.get_support()


## Bringing it all together
# You just joined an arrhythmia detection startup and want to train a model on the arrhythmias dataset arrh. You noticed that random forests tend to win quite a few Kaggle competitions, so you want to try that out with a maximum depth of 2, 5, or 10, using grid search. You also observe that the dimension of the dataset is quite high so you wish to consider the effect of a feature selection method.

# To make sure you don't overfit by mistake, you have already split your data. You will use X_train and y_train for the grid search, and X_test and y_test to decide if feature selection helps. All four dataset folds are preloaded in your environment. You also have access to GridSearchCV(), train_test_split(), SelectKBest(), chi2() and RandomForestClassifier as rfc.

# Instructions
# 100 XP
# Use grid search to experiment with a maximum depth of 2, 5, and 10 for RandomForestClassifier and store the best performing parameter setting.
# Now refit the estimator using the best-performing number of estimators as deduced above.
# Apply the SelectKBest feature selector with the chi2 scoring function and refit the classifier.

# Find the best value for max_depth among values 2, 5 and 10
grid_search = GridSearchCV(rfc(random_state=1), param_grid={'max_depth':[2,5,10]})
best_value = grid_search.fit(X_train, y_train).best_params_['max_depth']

# Using the best value from above, fit a random forest
clf = rfc(random_state=1, max_depth=best_value).fit(X_train, y_train)

# Apply SelectKBest with chi2 and pick top 100 features
vt = SelectKBest(chi2, k=100).fit(X_train, y_train)

# Create a new dataset only containing the selected features
# this returns an array of shape 339x100 (100 selected features)
X_train_reduced = vt.transform(X_train)





