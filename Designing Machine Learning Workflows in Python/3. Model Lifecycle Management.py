## Your first pipeline - again!
# Back in the arrhythmia startup, your monthly review is coming up, and as part of that an expert Python programmer will be reviewing your code. You decide to tidy up by following best practices and replace your script for feature selection and random forest classification, with a pipeline. You are using a training dataset available as X_train and y_train, and a number of modules: RandomForestClassifier, SelectKBest() and f_classif() for feature selection, as well as GridSearchCV and Pipeline.

# Instructions
# 100 XP
# Create a pipeline with the feature selector given by the sample code, and a random forest classifier. Name the first step feature_selection.
# Add two key-value pairs in params, one for the number of features k in the selector with values 10 and 20, and one for n_estimators in the forest with possible values 2 and 5.
# Initialize a GridSearchCV object with the given pipeline and parameter grid.
# Fit the object to the data and print the best performing parameter combination.

# Create pipeline with feature selector and classifier
pipe = Pipeline([
    ('feature_selection', SelectKBest(f_classif)),
    ('clf', RandomForestClassifier(random_state=2))])

# Create a parameter grid
params = {'feature_selection__k': [10,20],'clf__n_estimators': [2, 5]}

# Initialize the grid search object
grid_search = GridSearchCV(pipe, param_grid=params)

# Fit it to the data and print the best value combination
print(grid_search.fit(X_train, y_train).best_params_)

## Custom scorers in pipelines
# You are proud of the improvement in your code quality, but just remembered that previously you had to use a custom scoring metric in order to account for the fact that false positives are costlier to your startup than false negatives. You hence want to equip your pipeline with scorers other than accuracy, including roc_auc_score(), f1_score(), and you own custom scoring function. The pipeline from the previous lesson is available as pipe, as is the parameter grid as params and the training data as X_train, y_train. You also have confusion_matrix() for the purpose of writing your own metric.

# Instructions 1/3
# 35 XP
# Convert the metric roc_auc_score() into a scorer, and feed it into GridSearchCV(). Then fit that to the data.

# Create a custom scorer
scorer = make_scorer(roc_auc_score)

# Initialize the CV object
gs = GridSearchCV(pipe, param_grid=params, scoring=scorer)

# Fit it to the data and print the winning combination
print(gs.fit(X_train, y_train).best_params_)


# Now repeat for the F1 score, instead, given by f1_score().
# Create a custom scorer
scorer = make_scorer(f1_score)

# Initialise the CV object
gs = GridSearchCV(pipe, param_grid=params, scoring=scorer)

# Fit it to the data and print the winning combination
print(gs.fit(X_train, y_train).best_params_)

# Now repeat with a custom metric which is available to you as as simple Python function called my_metric().
# Create a custom scorer
scorer = make_scorer(my_metric)

# Initialise the CV object
gs = GridSearchCV(pipe, param_grid=params, scoring=scorer)

# Fit it to the data and print the winning combination
print(gs.fit(X_train, y_train).best_params_)

## Pickles
# Finally, it is time for you to push your first model to production. It is a random forest classifier which you will use as a baseline, while you are still working to develop a better alternative. You have access to the data split in training test with their usual names, X_train, X_test, y_train and y_test, as well as to the modules RandomForestClassifier() and pickle, whose methods .load() and .dump() you will need for this exercise.

# Instructions
# 100 XP
# Fit a random forest classifier to the data. Fix the random seed to 42 ensure that your results are reproducible.
# Write the model to file using pickle. Open the destination file using the with open(____) as ____ syntax.
# Now load the model from file into a different variable name, clf_from_file.
# Store the predictions from the model you loaded into a variable preds.

# Fit a random forest to the training set
clf = RandomForestClassifier(random_state=42).fit(
  X_train, y_train)

# Save it to a file, to be pushed to production
with open('model.pkl', 'wb') as file:
    pickle.dump(clf, file=file)

# Now load the model from file in the production environment
with open('model.pkl','rb') as file:
    clf_from_file = pickle.load(file)

# Predict the labels of the test dataset
preds = clf_from_file.predict(X_test)

## Custom function transformers in pipelines
# At some point, you were told that the sensors might be performing poorly for obese individuals. Previously you had dealt with that using weights, but now you are thinking that this information might also be useful for feature engineering, so you decide to replace the recorded weight of an individual with an indicator of whether they are obese. You want to do this using pipelines. You have numpy available as np, RandomForestClassifier(), FunctionTransformer(), and GridSearchCV().

# Instructions
# 100 XP
# Define a custom feature extractor. This is a function that will output a modified copy of its input.
# Replace each value of the first column with the indicator of whether that value is above a threshold given by a multiple of the column mean.
# Convert the feature extractor above to a transformer and place it in a pipeline together with a random forest classifier.
# Use grid search CV to try values 1, 2 and 3 for the multiplication constant multiplier in your feature extractor.

# Define a feature extractor to flag very large values
def more_than_average(X, multiplier=1.0):
  Z = X.copy()
  Z[:,1] = X[:,1] > multiplier*np.mean(Z[:,1])
  return Z

# Convert your function so that it can be used in a pipeline
pipe = Pipeline([('ft', FunctionTransformer(more_than_average)),('clf', RandomForestClassifier(random_state=2))])

# Optimize the parameter multiplier using GridSearchCV
# have to put 'ft__' in the beginning
params = {'ft__multiplier':[1,2,3]}
grid_search = GridSearchCV(pipe, param_grid=params)


## Challenge the champion
# Having pushed your random forest to production, you suddenly worry that a naive Bayes classifier might be better. You want to run a champion-challenger test, by comparing a naive Bayes, acting as the challenger, to exactly the model which is currently in production, which you will load from file to make sure there is no confusion. You will use the F1 score for assessment. You have the data X_train, X_test, y_train and y_test available as before and GaussianNB(), f1_score() and pickle().

# Instructions
# 100 XP
# Load the existing model from memory using pickle.
# Fit a Gaussian Naive Bayes classifier to the training data.
# Print the F1 score of the champion and then the challenger on the test data.
# Overwrite the current model to disk with the one that performed best.

# Load the current model from disk
champion = pickle.load(open('model.pkl', 'rb'))

# Fit a Gaussian Naive Bayes to the training data
challenger = GaussianNB().fit(X_train, y_train)

# Print the F1 test scores of both champion and challenger
print(f1_score(y_test, champion.predict(X_test)))
print(f1_score(y_test,challenger.predict(X_test)))

# Write back to disk the best-performing model
with open('model.pkl', 'wb') as file:
    pickle.dump(champion, file=file)
	
## Cross-validation statistics
# You used grid search CV to tune your random forest classifier, and now want to inspect the cross-validation results to ensure you did not overfit. In particular you would like to take the difference of the mean test score for each fold from the mean training score. The dataset is available as X_train and y_train, the pipeline as pipe, and a number of modules are pre-loaded including pandas as pd and GridSearchCV().

# Instructions
# 100 XP
# Create a grid search object with three cross-validation folds and ensure it returns training as well as test statistics.
# Fit the grid search object to the training data.
# Store the results of the cross-validation, available in the cv_results_ attribute of the fitted CV object, into a dataframe.
# Print the difference between the column containing the average test score and that containing the average training score.

# Fit your pipeline using GridSearchCV with three folds
grid_search = GridSearchCV(
  pipe, params, cv=3, return_train_score=True)

# Fit the grid search
gs = grid_search.fit(X_train, y_train)

# Store the results of CV into a pandas dataframe
results = pd.DataFrame(gs.cv_results_)

# Print the difference between mean test and training scores
print(
  results['mean_test_score']-results['mean_train_score'])
 
 
## Tuning the window size
# You want to check for yourself that the optimal window size for the arrhythmia dataset is 50. You have been given the dataset as a pandas data frame called arrh, and want to use a subset of the data up to time t_now. Your test data is available as X_test, y_test. You will try out a number of window sizes, ranging from 10 to 100, fit a naive Bayes classifier to each window, assess its F1 score on the test data, and then pick the best performing window size. You also have numpy available as np, and the function f1_score() has been imported already. Finally, an empty list called accuracies has been initialized for you to store the accuracies of the windows.

# Instructions
# 100 XP
# Define the index of a sliding window of size w_size stopping at t_now using the .loc() method.
# Construct X from the sliding window by removing the class column. Store that latter column as y.
# Fit a naive Bayes classifier to X and y, and use it to predict the labels of the test data X_test.
# Compute the F1 score of these predictions for each window size, and find the best-performing window size.

# Loop over window sizes
# Note: wrange is predefined as range(10,100,10)

for w_size in wrange:

    # Define sliding window
    # Refer to slide 38: (t_now-w_size+1:t_now) defines a Sliding window
    sliding = arrh.loc[(t_now - w_size + 1):t_now]

    # Extract X and y from the sliding window
    X, y = sliding.drop('class', 1), sliding['class']
    
    # Fit the classifier and store the F1 score
    preds = GaussianNB().fit(X, y).predict(X_test)
    accuracies.append(f1_score(y_test, preds))

# Estimate the best performing window size
# Note: wrange is pre-defined as wrange = range(10,100,10)
optimal_window = wrange[np.argmax(accuracies)]

## Bringing it all together
# You have two concerns about your pipeline at the arrhythmia detection startup:

# The app was trained on patients of all ages, but is primarily being used by fitness users who tend to be young. You suspect this might be a case of domain shift, and hence want to disregard all examples above 50 years old.
# You are still concerned about overfitting, so you want to see if making the random forest classifier less complex and selecting some features might help with that.
# You will create a pipeline with a feature selection SelectKBest() step and a RandomForestClassifier, both of which have been imported. You also have access to GridSearchCV(), Pipeline, numpy as np and pickle. The data is available as arrh.

# Instructions
# 100 XP
# Create a pipeline with SelectKBest() as step ft and RandomForestClassifier() as step clf.
# Create a parameter grid to tune k in SelectKBest() and max_depth in RandomForestClassifier().
# Use GridSearchCV() to optimize your pipeline against that grid and data containing only those aged under 50.
# Save the optimized pipeline to a pickle for production.

# Create a pipeline 
pipe = Pipeline([
  ('ft', SelectKBest()), ('clf', RandomForestClassifier(random_state=2))])

# Create a parameter grid
grid = {'ft__k':[5, 10], 'clf__max_depth':[10, 20]}

# Execute grid search CV on a dataset containing under 50s
grid_search = GridSearchCV(pipe, param_grid=grid)
arrh = arrh.iloc[np.where(arrh['age'] < 50)]
grid_search.fit(arrh.drop('class', 1), arrh['class'])

# Push the fitted pipeline to production
with open('pipe.pkl', 'wb') as file:
    pickle.dump(grid_search, file)
	
	


