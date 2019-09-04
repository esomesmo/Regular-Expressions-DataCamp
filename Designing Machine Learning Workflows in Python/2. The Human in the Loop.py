## Is the source or the destination bad?
# In the previous lesson, you used the destination computer as your entity of interest. However, your cybersecurity analyst just told you that it is the infected machines that generate the bad traffic, and will therefore appear as a source, not a destination, in the flows dataset.

# The data flows has been preloaded, as well as the list bad of infected IDs and the feature extractor featurizer() from the previous lesson. You also have numpy available as np, AdaBoostClassifier(), and cross_val_score().

# Instructions
# 100 XP
# Create a data frame where each row is a feature vector for a source_computer. Group by source computer ID in the flows dataset and apply the feature extractor to each group.
# Convert the iterator to a data frame by calling list() on it.
# Create labels by checking whether each source_computer ID belongs in the list of bads you have been given.
# Assess an AdaBoostClassifier() on this data using cross_val_score().

# Group by source computer, and apply the feature extractor
out = flows.groupby('source_computer').apply(featurize)

# Convert the iterator to a dataframe by calling list on it
X = pd.DataFrame(list(out), index=out.index)

# Check which sources in X.index are bad to create labels
y = [x in bads for x in X.index]

# Report the average accuracy of Adaboost over 3-fold CV
print(np.mean(cross_val_score(AdaBoostClassifier(), X, y)))

## Feature engineering on grouped data
# You will now build on the previous exercise, by considering one additional feature: the number of unique protocols used by each source computer. Note that with grouped data, it is always possible to construct features in this manner: you can take the number of unique elements of all categorical columns, and the mean of all numeric columns as your starting point. As before, you have flows preloaded, cross_val_score() for measuring accuracy, AdaBoostClassifier(), pandas as pd and numpy as np.

# Instructions
# 100 XP
# Apply a lambda function on the group iterator provided, to compute the number of unique protocols used by each source computer. You can use set() to reduce the protocol column to a set of unique values.
# Convert the result to a data frame with the right shape by providing an index and naming the column protocol.
# Concatenate the new data frame with the old one, which is available as X.
# Assess the accuracy of AdaBoostClassifier() on this new dataset using cross_val_score().

# Create a feature counting unique protocols per source
protocols = flows.groupby('source_computer').apply(lambda df: len(set(df['protocol'])))

# Convert this feature into a dataframe, naming the column
protocols_DF = pd.DataFrame(protocols, index=protocols.index, columns=['protocol'])

# Now concatenate this feature with the previous dataset, X
X_more = pd.concat([X, protocols_DF], axis=1)

# Refit the classifier and report its accuracy
print(np.mean(cross_val_score(AdaBoostClassifier(), X_more, y)))

## Turning a heuristic into a classifier
# You are surprised by the fact that heuristics can be so helpful. So you decide to treat the heuristic that "too many unique ports is suspicious" as a classifier in its own right. You achieve that by thresholding the number of unique ports per source by the average number used in bad source computers -- these are computers for which the label is True. The dataset is preloaded and split into training and test, so you have objects X_train, X_test, y_train and y_test in memory. Your imports include accuracy_score(), and numpy as np. To clarify: you won't be fitting a classifier from scikit-learn in this exercise, but instead you will define your own classification rule explicitly!

# Instructions
# 100 XP
# Subselect all bad hosts from X_train to form a new dataset X_train_bad. Note that y_train is a Boolean array.
# Calculate the average of the unique_ports column for bad hosts, and store it in avg_bad_ports.
# Now consider a classifier that predicts as positive every example whose unique_ports exceed avg_bad_ports. Save the predictions of this classifier on the test data on a new variable, pred_port.
# Calculate this classifier's accuracy on the test data using accuracy_score().

# Create a new dataset X_train_bad by subselecting bad hosts
X_train_bad = X_train[y_train]

# Calculate the average of unique_ports in bad examples
avg_bad_ports = np.mean(X_train_bad['unique_ports'])

# Label as positive sources that use more ports than that
pred_port = X_test['unique_ports'] > avg_bad_ports

# Print the accuracy of the heuristic
print(accuracy_score(y_test, pred_port))

## Combining heuristics
# A different cyber analyst tells you that during certain types of attack, the infected source computer sends small bits of traffic, to avoid detection. This makes you wonder whether it would be better to create a combined heuristic that simultaneously looks for large numbers of ports and small packet sizes. Does this improve performance over the simple port heuristic? As with the last exercise, you have X_train, X_test, y_train and y_test in memory. The sample code also helps you reproduce the outcome of the port heuristic, pred_port. You also have numpy as np and accuracy_score() preloaded.

# Instructions
# 100 XP
# The column average_packet computes the average packet size over all flows observed from a single source. Take the mean of those values for bad sources only on the training set.
# Now construct a new rule which flags as positive all sources whose average traffic is less than the value above.
# Combine the rules so that both heuristics have to simultaneously apply, using an appropriate arithmetic operation.
# Report the accuracy of the combined heuristic.

# Compute the mean of average_packet for bad sources
avg_bad_packet = np.mean(X_train[y_train]['average_packet'])

# Label as positive if average_packet is lower than that
pred_packet = X_test['average_packet'] < avg_bad_packet

# Find indices where pred_port and pred_packet both True
pred_port = X_test['unique_ports'] > avg_bad_ports
pred_both = pred_packet & pred_port

# Ports only produced an accuracy of 0.919. Is this better?
print(accuracy_score(y_test, pred_both))


## Dealing with label noise
# One of your cyber analysts informs you that many of the labels for the first 100 source computers in your training data might be wrong because of a database error. She hopes you can still use the data because most of the labels are still correct, but asks you to treat these 100 labels as "noisy". Thankfully you know how to do that, using weighted learning. The contaminated data is available in your workspace as X_train, X_test, y_train_noisy, y_test. You want to see if you can improve the performance of a GaussianNB() classifier using weighted learning. You can use the optional parameter sample_weight, which is supported by the .fit() methods of most popular classifiers. The function accuracy_score() is preloaded. You can consult the image below for guidance.



# Instructions
# 100 XP
# Fit an instance of GaussianNB() to the training data with the contaminated labels.
# Report its accuracy on the test data using accuracy_score().
# Create weights that assign twice as much weight to ground truth labels than to noisy labels. Remember that the weights concern the training data.
# Refit the classifier using the above weights and report its accuracy.

# Fit a Gaussian Naive Bayes classifier to the training data
clf = GaussianNB().fit(X_train, y_train_noisy)

# Report its accuracy on the test data
print(accuracy_score(y_test, clf.predict(X_test)))

# Assign half the weight to the first 100 noisy examples
weights = [0.5]*100 + [1.0]*(len(X_train)-100)

# Refit using weights and report accuracy. Has it improved?
clf_weights = GaussianNB().fit(X_train, y_train_noisy, sample_weight=weights)
print(accuracy_score(y_test, clf_weights.predict(X_test)))

## Reminder of performance metrics
# Remember the credit dataset? With all the extra knowledge you now have about metrics, let's have another look at how good a random forest is on this dataset. You have already trained your classifier and obtained your confusion matrix on the test data. The test data and the results are available to you as tp, fp, fn and tn, for true positives, false positives, false negatives, and true negatives respectively. You also have the ground truth labels for the test data, y_test and the predicted labels, preds. The functions f1_score() and precision_score() have also been imported.

# Instructions 1/3
# 35 XP
# Compute the F1 score for your classifier using the function f1_score().

print(f1_score(y_test, preds))

# Compute the precision for this classifier using the function precision_score().

print(precision_score(y_test, preds))

# Accuracy is the proportion of examples that were labelled correctly. Compute it without using accuracy_score().

print((tp + tn)/len(y_test))

## Real-world cost analysis
# You will still work on the credit dataset for this exercise. Recall that a "positive" in this dataset means "bad credit", i.e., a customer who defaulted on their loan, and a "negative" means a customer who continued to pay without problems. The bank manager informed you that the bank makes 10K profit on average from each "good risk" customer, but loses 150K from each "bad risk" customer. Your algorithm will be used to screen applicants, so those that are labeled as "negative" will be given a loan, and the "positive" ones will be turned down. What is the total cost of your classifier? The data is available as X_train, X_test, y_train and y_test. The functions confusion_matrix(), f1_score(), and precision_score() and RandomForestClassifier() are available.

# Instructions
# 100 XP
# Fit a random forest classifier to the training data.
# Use it to label the test data.
# Extract the false negatives and false positives from confusion_matrix(). You will have to flatten the matrix.
# Falsely classifying a "good" customer as "bad" means that the bank would have lost the chance to make 10K profit. Falsely classifying a "bad" customer as "good" means that the bank would have lost 150K due to the customer defaulting on their loan.

# Fit a random forest classifier to the training data
clf = RandomForestClassifier(random_state=2).fit(X_train, y_train)

# Label the test data
preds = clf.predict(X_test)

# Get false positives/negatives from the confusion matrix
tp, fp, fn, tn = confusion_matrix(y_test, preds).ravel()

# Now compute the cost using the manager's advice
cost = fp*10 + fn*150

## Default thresholding
# You would like to confirm that the DecisionTreeClassifier() uses the same default classification threshold as mentioned in the previous lesson, namely 0.5. It seems strange to you that all classifiers should use the same threshold. Let's check! A fitted decision tree classifier clf has been preloaded for you, as have the training and test data with their usual names: X_train, X_test, y_train and y_test. You will have to extract probability scores from the classifier using the .predict_proba() method.

# Instructions
# 100 XP
# Produce scores for the test examples, using the preloaded classifier clf.
# Now extract labels from the scores. Remember that you have a pair of scores for each example, not a single score, and the second element is the probability of the positive class.
# Now label the test data using the standard .predict() method
# Finally, compare with the predictions you got before. Are they identical?

# Score the test data using the given classifier
scores = clf.predict_proba(X_test)

# Get labels from the scores using the default threshold
# Remember: you have a pair of scores for each example, not a single score; and the second element is the probability of the positive class - this is why we use s[1]
preds = [s[1] > 0.5 for s in scores]

# Use the predict method to label the test data again
preds_default = clf.predict(X_test)

# Compare the two sets of predictions
all(preds == preds_default)

## Optimizing the threshold
# You heard that the default value of 0.5 maximizes accuracy in theory, but you want to test what happens in practice. So you try out a number of different threshold values, to see what accuracy you get, and hence determine the best-performing threshold value. You repeat this experiment for the F1 score. Is 0.5 the optimal threshold? Is the optimal threshold for accuracy and for the F1 score the same? Go ahead and find out! You have a scores matrix available, obtained by scoring the test data. The ground truth labels for the test data is also available as y_test. Finally, two numpy functions are preloaded, argmin() and argmax(), which retrieve the index of the minimum and maximum values in an array respectively, in addition to the metrics accuracy_score() and f1_score().

# Instructions
# 100 XP
# Create a range of threshold values that include 0.0, 0.25, 0.5, 0.75 and 1.0.
# Via double list comprehension, store the predictions for each threshold value in the range above. Recall that obtaining labels for a scores matrix using a threshold thr is possible using [s[1] > thr for s in scores].
# Run through that list and compute the accuracy for each threshold. Repeat for the F1 score.
# Using either argmin() or argmax(), find the optimal threshold for accuracy, and for F1.


# Create a range of equally spaced threshold values
t_range = [0.0,0.25,0.5,0.75,1.0]

# Store the predicted labels for each value of the threshold
preds = [[s[1] > thr for s in scores] for thr in t_range]

# Compute the accuracy for each threshold
accuracies = [accuracy_score(y_test,p) for p in preds]

# Compute the F1 score for each threshold
f1_scores = [f1_score(y_test, p) for p in preds]

# Report the optimal threshold for accuracy, and for F1
print(t_range[argmax(accuracies)], t_range[argmax(f1_scores)])

## Bringing it all together
# One of the engineers in your arrhythmia detection startup rushes into your office to let you know that there is a problem with the ECG sensor for overweight users. You decide to reduce the influence of examples with weight over 80 by 50%. You are also told that since your startup is targeting the fitness market and makes no medical claims, scaring an athlete unnecessarily is costlier than missing a possible case of arrhythmia. You decide to create a custom loss that makes each "false alarm" ten times costlier than missing a case of arrhythmia. Does down-weighting overweight subjects improve this custom loss? Your training data X_train, y_train and test data X_test, y_test are preloaded, as are confusion_matrix(), numpy as np, and DecisionTreeClassifier().

# Instructions 1/3
# 35 XP
# Start by creating a custom loss which extracts the false positives and false negatives from the confusion matrix, and then makes each false alarm count ten times as much as a missed case of arrhythmia.

# Create a scorer assigning more cost to false positives
def my_scorer(y_test, y_est, cost_fp=10.0, cost_fn=1.0):
    tn, fp, fn, tp = confusion_matrix(y_test,y_est).ravel()
    return fp*cost_fp + fn*cost_fn
	
# Fit a DecisionTreeClassifier to the original data and estimate this loss.

# Fit a DecisionTreeClassifier to the data and compute the loss
clf = DecisionTreeClassifier(random_state=2).fit(X_train,y_train)
print(my_scorer(y_test, clf.predict(X_test)))

# Create a list of weights so that each example where the weight is greater than 80 has half the weight of any other example. Does this improve your loss?


# Create a scorer assigning more cost to false positives
def my_scorer(y_test, y_est, cost_fp=10.0, cost_fn=1.0):
    tn, fp, fn, tp = confusion_matrix(y_test, y_est).ravel()
    return cost_fp*fp + cost_fn*fn

# Fit a DecisionTreeClassifier to the data and compute the loss
clf = DecisionTreeClassifier(random_state=2).fit(X_train, y_train)
print(my_scorer(y_test, clf.predict(X_test)))

# Refit, downweighting subjects whose weight is above 80
weights = [0.5 if w > 80 else 1.0 for w in X_train.weight]
clf_weighted = DecisionTreeClassifier(random_state=2).fit(
  X_train, y_train, sample_weight=weights)
print(my_scorer(y_test, clf_weighted.predict(X_test)))