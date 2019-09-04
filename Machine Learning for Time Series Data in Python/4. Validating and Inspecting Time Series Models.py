## Creating time-shifted features
# In machine learning for time series, it's common to use information about previous time points to predict a subsequent time point.

# In this exercise, you'll "shift" your raw data and visualize the results. You'll use the percent change time series that you calculated in the previous chapter, this time with a very short window. A short window is important because, in a real-world scenario, you want to predict the day-to-day fluctuations of a time series, not its change over a longer window of time.

# Instructions
# 100 XP
# Use a dictionary comprehension to create multiple time-shifted versions of prices_perc using the lags specified in shifts.
# Convert the result into a DataFrame.
# Use the given code to visualize the results.

# These are the "time lags"
shifts = np.arange(1, 11).astype(int)

# Use a dictionary comprehension to create name: value pairs, one pair per shift
shifted_data = {"lag_{}_day".format(day_shift): prices_perc.shift(day_shift) for day_shift in shifts}

# Convert into a DataFrame for subsequent use
prices_perc_shifted = pd.DataFrame(shifted_data)

# Plot the first 100 samples of each
ax = prices_perc_shifted.iloc[:100].plot(cmap=plt.cm.viridis)
prices_perc.iloc[:100].plot(color='r', lw=2)
ax.legend(loc='best')
plt.show()

## Special case: Auto-regressive models
# Now that you've created time-shifted versions of a single time series, you can fit an auto-regressive model. This is a regression model where the input features are time-shifted versions of the output time series data. You are using previous values of a timeseries to predict current values of the same timeseries (thus, it is auto-regressive).

# By investigating the coefficients of this model, you can explore any repetitive patterns that exist in a timeseries, and get an idea for how far in the past a data point is predictive of the future.

# Instructions
# 100 XP
# Replace missing values in prices_perc_shifted with the median of the DataFrame and assign it to X.
# Replace missing values in prices_perc with the median of the series and assign it to y.
# Fit a regression model using the X and y arrays.

# Replace missing values with the median for each column
X = prices_perc_shifted.fillna(np.nanmedian(prices_perc_shifted))
y = prices_perc.fillna(np.nanmedian(prices_perc))

# Fit the model
model = Ridge()
model.fit(X, y)

## Visualize regression coefficients
# Now that you've fit the model, let's visualize its coefficients. This is an important part of machine learning because it gives you an idea for how the different features of a model affect the outcome.

# The shifted time series DataFrame (prices_perc_shifted) and the regression model (model) are available in your workspace.

# In this exercise, you will create a function that, given a set of coefficients and feature names, visualizes the coefficient values.

# Instructions 1/2
# 80 XP
# Define a function (called visualize_coefficients) that takes as input an array of coefficients, an array of each coefficient's name, and an instance of a Matplotlib axis object. It should then generate a bar plot for the input coefficients, with their names on the x-axis.

def visualize_coefficients(coefs, names, ax):
    # Make a bar plot for the coefficients, including their names on the x-axis
    ax.bar(prices_perc_shifted.columns, coefs)
    ax.set(xlabel='Coefficient name', ylabel='Coefficient value')
    
    # Set formatting so it looks nice
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    return ax

# Use this function (visualize_coefficients()) with the coefficients contained in the model variable and column names of prices_perc_shifted.

# Visualize the output data up to "2011-01"
fig, axs = plt.subplots(2, 1, figsize=(10, 5))
y.loc[:'2011-01'].plot(ax=axs[0])

# Run the function to visualize model's coefficients
visualize_coefficients(model.coef_, prices_perc_shifted.columns, ax=axs[1])
plt.show()

## Auto-regression with a smoother time series
# Now, let's re-run the same procedure using a smoother signal. You'll use the same percent change algorithm as before, but this time use a much larger window (40 instead of 20). As the window grows, the difference between neighboring timepoints gets smaller, resulting in a smoother signal. What do you think this will do to the auto-regressive model?

# prices_perc_shifted and model (updated to use a window of 40) are available in your workspace.

# Instructions
# 100 XP
# Using the function (visualize_coefficients()) you created in the last exercise, generate a plot with coefficients of model and column names of prices_perc_shifted.

# Visualize the output data up to "2011-01"
fig, axs = plt.subplots(2, 1, figsize=(10, 5))
y.loc[:'2011-01'].plot(ax=axs[0])

# Run the function to visualize model's coefficients
visualize_coefficients(model.coef_, prices_perc_shifted.columns, ax=axs[1])
plt.show()



## Cross-validation with shuffling
# As you'll recall, cross-validation is the process of splitting your data into training and test sets multiple times. Each time you do this, you choose a different training and test set. In this exercise, you'll perform a traditional ShuffleSplit cross-validation on the company value data from earlier. Later we'll cover what changes need to be made for time series data. The data we'll use is the same historical price data for several large companies.

# An instance of the Linear regression object (model) is available in your workspace along with the function r2_score() for scoring. Also, the data is stored in arrays X and y. We've also provided a helper function (visualize_predictions()) to help visualize the results.

# Instructions
# 100 XP
# Initialize a ShuffleSplit cross-validation object with 10 splits.
# Iterate through CV splits using this object. On each iteration:
# Fit a model using the training indices.
# Generate predictions using the test indices, score the model (R2) using the predictions, and collect the results.

# Import ShuffleSplit and create the cross-validation object
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10, random_state=1)

# Iterate through CV splits
results = []
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    model.fit(X[tr], y[tr])
    
    # Generate predictions on the test data, score the predictions, and collect
    prediction = model.predict(X[tt])
    score = r2_score(y[tt],prediction )
    results.append((prediction, score, tt))

# Custom function to quickly visualize predictions
visualize_predictions(results)



## Cross-validation without shuffling
# Now, re-run your model fit using block cross-validation (without shuffling all datapoints). In this case, neighboring time-points will be kept close to one another. How do you think the model predictions will look in each cross-validation loop?

# An instance of the Linear regression model object is available in your workspace. Also, the arrays X and y (training data) are available too.

# Instructions
# 100 XP
# Instantiate another cross-validation object, this time using KFold cross-validation with 10 splits and no shuffling.
# Iterate through this object to fit a model using the training indices and generate predictions using the test indices.
# Visualize the predictions across CV splits using the helper function (visualize_predictions()) we've provided.

# Create KFold cross-validation object
from sklearn.model_selection import KFold
cv = KFold(n_splits=10, shuffle=False, random_state=1)

# Iterate through CV splits
results = []
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    model.fit(X[tr],y[tr])
    
    # Generate predictions on the test data and collect
    prediction = model.predict(X[tt])
    results.append((prediction, tt))
    
# Custom function to quickly visualize predictions
visualize_predictions(results)


## Time-based cross-validation
# Finally, let's visualize the behavior of the time series cross-validation iterator in scikit-learn. Use this object to iterate through your data one last time, visualizing the training data used to fit the model on each iteration.

# An instance of the Linear regression model object is available in your workpsace. Also, the arrays X and y (training data) are available too.

# Instructions
# 100 XP
# Import TimeSeriesSplit from sklearn.model_selection.
# Instantiate a time series cross-validation iterator with 10 splits.
# Iterate through CV splits. On each iteration, visualize the values of the input data that would be used to train the model for that iteration.

# Import TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit

# Create time-series cross-validation object
cv = TimeSeriesSplit(n_splits=10)

# Iterate through CV splits
fig, ax = plt.subplots()
for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Plot the training data on each iteration, to see the behavior of the CV
    ax.plot(tr, ii + y[tr])

ax.set(title='Training data on each CV iteration', ylabel='CV iteration')
plt.show()


## Bootstrapping a confidence interval
# A useful tool for assessing the variability of some data is the bootstrap. In this exercise, you'll write your own bootstrapping function that can be used to return a bootstrapped confidence interval.

# This function takes three parameters: a 2-D array of numbers (data), a list of percentiles to calculate (percentiles), and the number of boostrap iterations to use (n_boots). It uses the resample function to generate a bootstrap sample, and then repeats this many times to calculate the confidence interval.

# Instructions
# 100 XP
# The function should loop over the number of bootstraps (given by the parameter n_boots) and:
# Take a random sample of the data, with replacement, and calculate the mean of this random sample
# Compute the percentiles of bootstrap_means and return it

from sklearn.utils import resample

def bootstrap_interval(data, percentiles=(2.5, 97.5), n_boots=100):
    """Bootstrap a confidence interval for the mean of columns of a 2-D dataset."""
    # Create our empty array to fill the results
    bootstrap_means = np.zeros([n_boots, data.shape[-1]])
    for ii in range(n_boots):
        # Generate random indices for our data *with* replacement, then take the sample mean
        random_sample = resample(data)
        bootstrap_means[ii] = random_sample.mean(axis=0)
        
    # Compute the percentiles of choice for the bootstrapped means
    percentiles = np.percentile(bootstrap_means, percentiles, axis=0)
    return percentiles


## Calculating variability in model coefficients
# In this lesson, you'll re-run the cross-validation routine used before, but this time paying attention to the model's stability over time. You'll investigate the coefficients of the model, as well as the uncertainty in its predictions.

# Begin by assessing the stability (or uncertainty) of a model's coefficients across multiple CV splits. Remember, the coefficients are a reflection of the pattern that your model has found in the data.

# An instance of the Linear regression object (model) is available in your workpsace. Also, the arrays X and y (the data) are available too.

# Instructions 1/2
# 0 XP
# Initialize a TimeSeriesSplit cross-validation object
# Create an array of all zeros to collect the coefficients.
# Iterate through splits of the cross-validation object. On each iteration:
# Fit the model on training data
# Collect the model's coefficients for analysis later

# Iterate through CV splits
n_splits = 100
cv = TimeSeriesSplit(n_splits=n_splits)

# Create empty array to collect coefficients
coefficients = np.zeros([n_splits, X.shape[1]])

for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Fit the model on training data and collect the coefficients
    model.fit(X[tr], y[tr])
    coefficients[ii] = model.coef_

# Finally, calculate the 95% confidence interval for each coefficient in coefficients using the bootstrap_interval() function you defined in the previous exercise. You can run bootstrap_interval? if you want a refresher on the parameters that this function takes.

# Calculate a confidence interval around each coefficient
bootstrapped_interval = bootstrap_interval(coefficients,percentiles=(2.5,97.5),n_boots=100)

# Plot it
fig, ax = plt.subplots()
ax.scatter(feature_names, bootstrapped_interval[0], marker='_', lw=3)
ax.scatter(feature_names, bootstrapped_interval[1], marker='_', lw=3)
ax.set(title='95% confidence interval for model coefficients')
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()


## Visualizing model score variability over time
# Now that you've assessed the variability of each coefficient, let's do the same for the performance (scores) of the model. Recall that the TimeSeriesSplit object will use successively-later indices for each test set. This means that you can treat the scores of your validation as a time series. You can visualize this over time in order to see how the model's performance changes over time.

# An instance of the Linear regression model object is stored in model, a cross-validation object in cv, and data in X and y.

# Instructions 1/2
# 90 XP
# Calculate the cross-validated scores of the model on the data (using a custom scorer we defined for you, my_pearsonr along with cross_val_score).
# Convert the output scores into a pandas Series so that you can treat it as a time series.
# Bootstrap a rolling confidence interval for the mean score using bootstrap_interval().

from sklearn.model_selection import cross_val_score

# Generate scores for each split to see how the model performs over time
scores = cross_val_score(model, X, y, cv=cv, scoring=my_pearsonr)

# Convert to a Pandas Series object
# time_scores is of dtype: datetime64[ns]
# this is of length 100
scores_series = pd.Series(scores, index=times_scores, name='score')

# Bootstrap a rolling confidence interval for the mean score
scores_lo = scores_series.rolling(20).aggregate(partial(bootstrap_interval, percentiles=2.5))
scores_hi = scores_series.rolling(20).aggregate(partial(bootstrap_interval, percentiles=97.5))

# Run the given code to plot the results.
# Plot the results
fig, ax = plt.subplots()
scores_lo.plot(ax=ax, label="Lower confidence interval")
scores_hi.plot(ax=ax, label="Upper confidence interval")
ax.legend()
plt.show()

## Accounting for non-stationarity
# In this exercise, you will again visualize the variations in model scores, but now for data that changes its statistics over time.

# An instance of the Linear regression model object is stored in model, a cross-validation object in cv, and the data in X and y.

# Instructions 1/2
# 63 XP
# Create an empty DataFrame to collect the results.
# Iterate through multiple window sizes, each time creating a new TimeSeriesSplit object.
# Calculate the cross-validated scores (using a custom scorer we defined for you, my_pearsonr) of the model on training data.

# Pre-initialize window sizes
window_sizes = [25, 50, 75, 100]

# Create an empty DataFrame to collect the stores
all_scores = pd.DataFrame(index=times_scores)

# Generate scores for each split to see how the model performs over time
for window in window_sizes:
    # Create cross-validation object using a limited lookback window
    cv = TimeSeriesSplit(n_splits=100, max_train_size=window)
    
    # Calculate scores across all CV splits and collect them in a DataFrame
    this_scores = cross_val_score(model, X, y, cv=cv, scoring=my_pearsonr)
    all_scores['Length {}'.format(window)] = this_scores

# Visualize the scores
ax = all_scores.rolling(10).mean().plot(cmap=plt.cm.coolwarm)
ax.set(title='Scores for multiple windows', ylabel='Correlation (r)')
plt.show()


