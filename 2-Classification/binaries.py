from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline') # the output of plotting commands is displayed inline within frontends like the Jupyter notebook, directly below the code cell that produced it
import numpy   as np
import pandas  as pd
from sklearn.tree    import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#imports for the Decision Tree
from matplotlib import pyplot
from sklearn.tree import plot_tree
from matplotlib.pyplot import figure

in_train = "Data/binaries_train.csv"
in_test  = "Data/binaries_test.csv"

# Read the in_train file into the train dataframe and inspect it
train = pd.read_csv(in_train, sep=',')
test  = pd.read_csv(in_test,  sep=',')

train.head()

# Preparing features and target
# Store in X the content of iris excluding the column species and verify the shape 
X_train = train.drop(columns=['Class'])
np.shape(X_train) # (64,6)
y_train = train['Class']
np.shape(y_train) # (64,)


# --- TRAIN A FULL TREE ---
# Choose the model class DecisionTreeClassifier and instantiate it
dtc = DecisionTreeClassifier(class_weight=None, criterion='entropy', min_samples_leaf=1, min_samples_split=2, splitter='best',min_weight_fraction_leaf=0.0)
dtc.fit(X_train, y_train)

# use the trained model to predict y_predicted_train from X_train.
ytrain_model = dtc.predict(X_train)
print("y_train mean: {}".format(np.mean(y_train)))
print("ytrain_model mean: {}".format(np.mean(ytrain_model)))
print("The accuracy on training set is {}%".format(accuracy_score(y_train, ytrain_model)*100)) # 100%

# load the test set, make the prediction using the already trained model and compute the accuracy
Xtest = test.drop(columns=['Class'])
ytest = test['Class']
ytest_model = dtc.predict(Xtest)
print("The accuracy on training set is {}%".format(accuracy_score(ytest, ytest_model)*100)) #60.9% :((


# --- OBSERVE THE TREE ---
figure(figsize = (25,25))
#print(X_train.columns)
plot_tree(dtc, max_depth = 2, filled=True, rounded=True, feature_names=X_train.columns, class_names=['False','True'])

dtc_pruned = DecisionTreeClassifier(class_weight=None, criterion='entropy',max_depth=2 ,min_samples_leaf=1, min_samples_split=2, splitter='best',min_weight_fraction_leaf=0.0)
dtc_pruned.fit(X_train, y_train)
ytrain_pruned_model = dtc_pruned.predict(X_train)
print("The accuracy on training set is {}%".format(accuracy_score(y_train, ytrain_pruned_model)*100)) # 81%
ytest_pruned_model = dtc_pruned.predict(Xtest)
print("The accuracy on training set is {}%".format(accuracy_score(ytest, ytest_pruned_model)*100)) #60.9% :((