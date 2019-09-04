## A simple outlier
# When you first encounter a new type of algorithm, it is always a great idea to test it with a very simple example. So you decide to create a list containing thirty examples with the value 1.0 and just one example with value 10.0, which you expect should be flagged as an outlier. To make sure you use the algorithm correctly, you convert the list to a pandas dataframe, and feed it into the local outlier factor algorithm. pandas is available to you as pd.

# Instructions
# 100 XP
# Import the LocalOutlierFactor module as lof for convenience.
# Create a list with thirty 1s followed by a 10, [1.0, 1.0, ..., 1.0, 10.0].
# Cast the list to a data frame.
# Print the outlier scores produced by the local outlier factor algorithm.

# Import the LocalOutlierFactor module
from sklearn.neighbors import LocalOutlierFactor as lof

# Create the list [1.0, 1.0, ..., 1.0, 10.0] as explained
x = [1.0]*30
x.append(10.0)

# Cast to a data frame
X = pd.DataFrame(x)

# Fit the local outlier factor and print the outlier scores
print(lof().fit_predict(X))

## LoF contamination
# Your medical advisor at the arrhythmia startup informs you that your training data might not contain all possible types of arrhythmia. How on earth will you detect these other types without any labeled examples? Could an anomaly detector tell the difference between healthy and unhealthy without access to labels? But first, you experiment with the contamination parameter to see its effect on the confusion matrix. You have LocalOutlierFactor as lof, numpy as np, the labels as ground_truth encoded in -1and 1 just like local outlier factor output, and the unlabeled training data as X.

# Instructions 1/3
# 35 XP
# Fit a local outlier factor and output the predictions on X and print the confusion matrix for these predictions.

# Fit the local outlier factor and output predictions
preds = lof().fit_predict(X)

# Print the confusion matrix
print(confusion_matrix(ground_truth, preds))

# Repeat but now set the proportion of datapoints to be flagged as outliers to 0.2. Print the confusion matrix.

# Set the contamination parameter to 0.2
preds = lof(contamination=0.2).fit_predict(X)

# Print the confusion matrix
print(confusion_matrix(ground_truth, preds))

# Now set the contamination to be equal to the actual proportion of outliers in the data.

# Contamination to match outlier frequency in ground_truth
preds = lof(
  contamination=np.mean(ground_truth==-1.0)).fit_predict(X)

# Print the confusion matrix
print(confusion_matrix(ground_truth, preds))

## A simple novelty
# You find novelty detection more useful than outlier detection, but want to make sure it works on the simple example you came up with before. This time you will use a sequence of thirty examples all with value 1.0 as a training set, and try to see if the example 10.0 is labeled as a novelty. You have access to pandas as pd, and the LocalOutlierFactor module as lof.

# Instructions
# 100 XP
# Create a pandas DataFrame containing thirty examples all equal to 1.0.
# Initialize a local outlier factor novelty detector.
# Fit the detector to the training data.
# Output the novelty label of the datapoint 10.0, casted to a DataFrame.

# Create a list of thirty 1s and cast to a dataframe
X = pd.DataFrame([1.0]*30)

# Create an instance of a lof novelty detector
detector = lof(novelty=True)

# Fit the detector to the data
detector.fit(X)

# Use it to predict the label of an example with value 10.0
print(detector.predict(pd.DataFrame([10.0])))

## Three novelty detectors
# Finally, you know enough to run some tests on the use of a few anomaly detectors on the arrhythmia dataset. To test their performance, you will train them on an unlabeled training dataset, but then compare their predictions to the ground truth on the test data using their method .score_samples(). This time, you will be asked to import the detectors as part of the exercise, but you do have the data X_train, X_test, y_train, y_test preloaded as usual.

# Instructions 1/3
# 35 XP
# Import the one-class SVM detector from the svm module as onesvm, fit it to the training data, and score the test data.

# Import the novelty detector
from sklearn.svm import OneClassSVM as onesvm

# Fit it to the training data and score the test data
svm_detector = onesvm().fit(X_train)
scores = svm_detector.score_samples(X_test)

# Adapt your code to import the isolation forest from the ensemble module as isof, fit it and score the test data.

# Import the novelty detector
from sklearn.ensemble import IsolationForest as isof

# Fit it to the training data and score the test data
svm_detector = isof().fit(X_train)
scores = svm_detector.score_samples(X_test)

# Adapt your code to import the LocalOutlierFactor module as lof, fit it to the training data, and score the test data.

# Import the novelty detector
from sklearn.neighbors import LocalOutlierFactor as lof

# Fit it to the training data and score the test data
lof_detector = lof(novelty=True).fit(X_train)
scores = lof_detector.score_samples(X_test)

## Contamination revisited
# You notice that one-class SVM does not have a contamination parameter. But you know well by now that you really need a way to control the proportion of examples that are labeled as novelties in order to control your false positive rate. So you decide to experiment with thresholding the scores. The detector has been imported as onesvm, you also have available the data as X_train, X_test, y_train, y_test, numpy as np, and confusion_matrix().

# Instructions
# 100 XP
# Fit the 1-class SVM and score the test data.
# Compute the observed proportion of outliers in the test data.
# Use np.quantile() to find where to threshold the scores to achieve that proportion.
# Use that threshold to label the test data. Print the confusion matrix.

# Fit a one-class SVM detector and score the test data
nov_det = onesvm().fit(X_train)
scores = nov_det.score_samples(X_test)

# Find the observed proportion of outliers in the test data
prop = np.mean(y_test==1.0)

# Compute the appropriate threshold
threshold = np.quantile(scores, prop)

# Print the confusion matrix for the thresholded scores
print(confusion_matrix(y_test, scores > threshold))


## Find the neighbor
# It is clear that the local outlier factor algorithm depends a lot on the idea of a nearest neighbor, which in turn depends on the choice of distance metric. So you decide to experiment some more with the hepatitis dataset introduced in the previous lesson. You are given three examples stored in features, whose classes are stored in labels. You will identify the nearest neighbor to the first example (row with index 0) using three different distance metrics, Euclidean, Hamming and Chebyshev, and on the basis of that choose which distance metric to use. You will import the necessary module as part of the exercise, but pandas and numpy already available, as are features and their labels labels.

# Import DistanceMetric as dm
from sklearn.neighbors import DistanceMetric as dm

# Find the Euclidean distance between all pairs
dist_eucl = dm.get_metric('euclidean').pairwise(features)

# Find the Hamming distance between all pairs
dist_hamm = dm.get_metric('hamming').pairwise(features)

# Find the Chebyshev distance between all pairs
dist_cheb = dm.get_metric('chebyshev').pairwise(features)

## Not all metrics agree
# In the previous exercise you saw that not all metrics agree when it comes to identifying nearest neighbors. But does this mean they might disagree on outliers, too? You decide to put this to the test. You use the same data as before, but this time feed it into a local outlier factor outlier detector. The module LocalOutlierFactor has been made available to you as lof, and the data is available as features.

# Instructions
# 100 XP
# Detect outliers in features using the euclidean metric.
# Detect outliers in features using the hamming metric.
# Detect outliers in features using the jaccard metric.
# Find if all three metrics agree on any one outlier.

# Compute outliers according to the euclidean metric
out_eucl = lof(metric='euclidean').fit_predict(features)

# Compute outliers according to the hamming metric
out_hamm = lof(metric='hamming').fit_predict(features)

# Compute outliers according to the jaccard metric
out_jacc  = lof(metric='jaccard').fit_predict(features)

# Find if the metrics agree on any one datapoint
# the score of an outlier is -1, so if all metrics agree on one examp,e its cumulative score would be -3
print(any(out_eucl + out_hamm + out_jacc == -3))

## Restricted Levenshtein
# You notice that the stringdist package also implements a variation of Levenshtein distance called the Restricted Damerau-Levenshtein distance, and want to try it out. You will follow the logic from the lesson, wrapping it inside a custom function and precomputing the distance matrix before fitting a local outlier factor anomaly detector. You will measure performance with accuracy_score() which is available to you as accuracy(). You also have access to packages stringdist, numpy as np, pdist() and squareform() from scipy.spatial.distance, and LocalOutlierFactor as lof. The data has been preloaded as a pandas dataframe with two columns, label and sequence, and has two classes: IMMUNE SYSTEM and VIRUS.

# Instructions
# 100 XP
# Write a function with input u and v, each of which is an array containing a string, and applies the rdlevenshtein() function on the two strings.
# Reshape the sequence column from proteins by first casting it into an numpy array, and then using .reshape().
# Compute a square distance matrix for sequences using my_rdlevenshtein(), and fit lof on it.
# Compute accuracy by converting preds and proteins['label'] into booleans indicating whether a protein is a virus.

# Wrap the RD-Levenshtein metric in a custom function
def my_rdlevenshtein(u, v):
    return stringdist.rdlevenshtein(u[0], v[0])

# Reshape the array into a numpy matrix
sequences = np.array(proteins['seq']).reshape(-1, 1)

# Compute the pairwise distance matrix in square form
M = squareform(pdist(sequences,my_rdlevenshtein))

# Run a LoF algorithm on the precomputed distance matrix
preds = lof(metric='precomputed').fit_predict(M)

# Compute the accuracy of the outlier predictions
# See slide 33
print(accuracy(proteins['label']=='VIRUS', preds==-1))

## Bringing it all together
# In addition to the distance-based learning anomaly detection pipeline you created in the last exercise, you want to also support a feature-based learning one with one-class SVM. You decide to extract two features: first, the length of the string, and, second, a numerical encoding of the first letter of the string, obtained using the function LabelEncoder() described in Chapter 1. To ensure a fair comparison, you will input the outlier scores into an AUC calculation. The following have been imported: LabelEncoder(), roc_auc_score() as auc() and OneClassSVM. The data is available as a pandas data frame called proteins with two columns, label and seq, and two classes, IMMUNE SYSTEM and VIRUS. A fitted LoF detector is available as lof_detector.

# Instructions
# 100 XP
# For a string s, len(s) returns its length. Apply that to the seq column to obtain a new column len.
# For a string s, list(s) returns a list of its characters. Use this to extract the first letter of each sequence, and encode it using LabelEncoder().
# LoF scores are in the negative_outlier_factor_ attribute. Compute their AUC.
# Fit a 1-class SVM to a data frame with only len and first as columns. Extract the scores and assess both the LoF scores and the SVM scores using AUC.

# Create a feature that contains the length of the string
proteins['len'] = proteins['seq'].apply(lambda s: len(s))

# Create a feature encoding the first letter of the string
proteins['first'] =  LabelEncoder().fit_transform(proteins['seq'].apply(lambda s: list(s)[0]))

# Extract scores from the fitted LoF object, compute its AUC
scores_lof = lof_detector.negative_outlier_factor_
print(auc(proteins['label']=='IMMUNE SYSTEM', scores_lof))

# Fit a 1-class SVM, extract its scores, and compute its AUC
svm = OneClassSVM().fit(proteins[['len', 'first']])
scores_svm = svm.score_samples(proteins[['len', 'first']])
print(auc(proteins['label']=='IMMUNE SYSTEM', scores_svm))


