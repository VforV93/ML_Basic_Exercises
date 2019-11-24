from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline') # the output of plotting commands is displayed inline within frontends like the Jupyter notebook, directly below the code cell that produced it
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from matplotlib.pyplot import figure
from sklearn.tree import plot_tree

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


data_url    = "Data/winequality-red.csv"
target_name = 'quality'


plt.rcParams['figure.figsize'] = [20, 20]
random_state = 15
np.random.seed(random_state)

# Read data from file 'filename.csv' 
df = pd.read_csv( data_url , sep=';')
print("Shape of the input data {}".format(df.shape))

# Show column names 
df.columns

# Show portion of data
df.head()

#Use the hist method of the DataFrame to show the histograms of the attributes
df.hist()

#Print the unique class labels (hint: use the unique method of pandas Series)
classes = df[ target_name ].unique()
classes.sort()
print(classes)

# Preparing features and target
X = df.drop(columns=[ target_name ]) # drop the Class column
np.shape(X) # (1599,11)
y = df[ target_name ] # Class only
np.shape(y) # (1599,)



# -- Prepare a simple model selection: HOLDOUT METHOD -- #

# Generate the variables Xtrain, Xtest, ytrain, ytest
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=random_state) # default Train 0.75- Test 0.25
# Show the shape of the resulting variables
np.shape(Xtrain) # (1199,11)
np.shape(Xtest)  # (400,11)
np.shape(ytrain) # (1199,)
np.shape(ytest)  # (400,)
print("There are {} samples in the training dataset".format(Xtrain.shape[0]))
print("There are {} samples in the testing dataset".format(Xtest.shape[0]))
print("Each sample has {} features".format(Xtrain.shape[1]))


# PART 1
# Initialise an estimator with the chosen model generato
estimator = tree.DecisionTreeClassifier(criterion="entropy")
estimator.fit(Xtrain, ytrain)

# --- SHOW THE DECISION TREE ---
figure(figsize=[10, 10])
plot_tree(estimator, filled=True, feature_names=X.columns, 
class_names=str(estimator.classes_), rotate=False, max_depth = 1)
# --- --- --- --- --- --- --- ---

# Part 1.1
# predict the y values with the ﬁtted estimator and the train data
# compare the predicted values with the true ones and compute the accuracy on the training set
ytrain_model = estimator.predict(Xtrain)
print("The accuracy on training set is {}%".format(accuracy_score(ytrain, ytrain_model)*100))

ytest_model = estimator.predict(Xtest)
print("The accuracy on test set is {}%".format(accuracy_score(ytest, ytest_model)*100))
max_depth = estimator.tree_.max_depth
print("The maximum depth of the fitted tree is {}".format(max_depth))


# PART 2 - HOLDOUT METHOD
# Optimising the tree: limit the maximum tree depth. We will use the three way splitting: train, validation, test.
# split the training set into two parts: train_t and val
Xtrain_t, Xval, ytrain_t, yval = train_test_split(Xtrain, ytrain, random_state=random_state) # default Train 0.75- Test 0.25
print("There are {} samples in the training dataset".format(Xtrain_t.shape[0]))
print("There are {} samples in the validation dataset".format(Xval.shape[0]))

# Loop for computing the score varying the hyperparameter
scores = []

parameter_values = [exp for exp in np.arange(1,max_depth+1, 1)]
for par in parameter_values:
    estimator = tree.DecisionTreeClassifier(criterion="entropy", max_depth = par)
    estimator.fit(Xtrain_t, ytrain_t)
    ypredicted_val = estimator.predict(Xval)
    score =  accuracy_score(yval, ypredicted_val) * 100 # compute the matches between prediction and true classes
    scores.append(score)

# Plot the results
plt.figure(figsize=(32,20))
plt.plot(parameter_values, scores, '-o', linewidth=5, markersize=24)
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.title("Score with validation varying max_depth of tree", fontsize = 24)
plt.show();

top_par_hov = parameter_values[np.argmax(scores)]
best_estimator = tree.DecisionTreeClassifier(criterion="entropy", max_depth = top_par_hov)
best_estimator.fit(Xtrain_t, ytrain_t)
ypredicted_test = estimator.predict(Xtest)
accuracy_hov = accuracy_score(ypredicted_test, ytest) * 100
print("The top accuracy is {0:.1f}%".format(accuracy_hov))
print("Obtained with max_depth = {}".format(top_par_hov))


#  PART 3 - TUNING WITH CROSS VALIDATION
# Optimisation of the hyperparameter with cross validation
# Now we will tune the hyperparameter looping on cross validation with the training set, then 
# we will fit the estimator on the training set and evaluate the performance on the test set
avg_scores = []
for par in parameter_values:
    estimator = tree.DecisionTreeClassifier( criterion = "entropy", 
                                             max_depth = par )
    scores = cross_val_score( estimator, Xtrain, ytrain,
                              scoring = 'accuracy', 
                              cv = 5 )
    # cross_val_score produces an array with one score for each fold
    avg_scores.append(np.mean(scores))
print(avg_scores)

# Plot using the parameter_values and the list of scores
plt.figure(figsize=(32,20))
plt.plot(parameter_values, avg_scores, '-o', linewidth=5, markersize=24)
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.title("Score with Cross Validation varying max_depth of tree", fontsize = 24)
plt.show();

# Fit the tree after cross validation and print summary
top_par_cv = parameter_values[np.argmax(avg_scores)]
best_estimator_cv = tree.DecisionTreeClassifier( criterion="entropy", max_depth = top_par_cv )
best_estimator_cv.fit(Xtrain_t, ytrain_t)
ypredicted_test = best_estimator_cv.predict(Xtest)
accuracy_cv = accuracy_score(ypredicted_test, ytest) * 100
print("The top accuracy is {0:.1f}%".format(accuracy_cv))
print("Obtained with max_depth = {}".format(top_par_cv))


print(classification_report(ytest, ypredicted_test))

print(confusion_matrix(ytest, ypredicted_test))



#  PART 4 - TUNING WITH AN ENSEMBLE METHOD
scores_bagging = []
for par in parameter_values:
    estimator_bagging = tree.BaggingClassifier(tree.DecisionTreeClassifier(criterion="entropy"
                                            , max_depth = par
                                            )
                                          , max_samples=0.5
                                          , max_features=0.5
                                         )
    estimator_bagging.fit(Xtrain,ytrain)
    scores = cross_val_score(estimator_bagging, Xtrain, ytrain
                             , scoring='accuracy', cv = 5
                            )
    scores_bagging.append(np.mean(scores))
print(scores_bagging)