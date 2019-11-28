"""
http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
@author: scikit-learn.org and Claudio Sartori
"""
import warnings
warnings.filterwarnings('ignore') # uncomment this line to suppress warnings

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
print(__doc__) # print information included in the triple quotes at the beginning

# Loading a standard dataset
#dataset = datasets.load_digits()
#dataset = datasets.fetch_olivetti_faces()
#dataset = datasets.fetch_covtype()
dataset = datasets.load_iris()
#dataset = datasets.load_wine()
#dataset = datasets.load_breast_cancer()

df = pd.DataFrame(data= np.c_[dataset['data'], dataset['target']], columns= dataset['feature_names'] + ['target'])
print(df.head())
print()

# Prepare the environment
ts = 0.3
random_state = 15
X = dataset.data
np.shape(X) # (150,5)
y = dataset.target
np.shape(y) # (150,)



# Generate the variables Xtrain, Xtest, ytrain, ytest
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=random_state, test_size=ts) # default Train 0.75- Test 0.25
# Show the shape of the resulting variables
print("Xtrain shape is: {}".format(np.shape(Xtrain))) # (105,4)
print("Xtest  shape is: {}".format(np.shape(Xtest)))  # (45,4)
print("ytrain shape is: {}".format(np.shape(ytrain))) # (105,)
print("ytest  shape is: {}".format(np.shape(ytest)))  # (45,)

model_lbls = [
#              'dt', 
#              'nb', 
#              'lp', 
              'svc', 
#             'knn',
            ]

# Set the parameters by cross-validation
tuned_param_dt = [{'max_depth': [range(1,20)]}]
tuned_param_nb = [{'var_smoothing': [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-07, 1e-8, 1e-9, 1e-10]}]
tuned_param_lp = [{'early_stopping': [True]}]
tuned_param_svc = [{'kernel': ['rbf'], 
                    'gamma': [1e-3, 1e-4],
                    'C': [1, 10, 100, 1000],
                    },
                    {'kernel': ['linear'],
                     'C': [1, 10, 100, 1000],                     
                    },
                   ]
tuned_param_knn =[{'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]

models = {
    'dt': {'name': 'Decision Tree       ',
           'estimator': DecisionTreeClassifier(), 
           'param': tuned_param_dt,
          },
    'nb': {'name': 'Gaussian Naive Bayes',
           'estimator': GaussianNB(),
           'param': tuned_param_nb
          },
    'lp': {'name': 'Linear Perceptron   ',
           'estimator': Perceptron(),
           'param': tuned_param_lp,
          },
    'svc':{'name': 'Support Vector      ',
           'estimator': SVC(), 
           'param': tuned_param_svc
          },
    'knn':{'name': 'K Nearest Neighbor ',
           'estimator': KNeighborsClassifier(),
           'param': tuned_param_knn
        
    }
}

scores = ['precision', 'recall']

# The function below groups all the outputs
# Write a function which has as parameter the fitted model and uses the components of the fitted model to inspect the results of the search with the parameters grid.
# Components:
# model.best_params_
# model.cv_results_['mean_test_score']
# model.cv_results_['std_test_score']
# model.cv_results_['params']

def print_results(model):
    print("Best parameters set found on train set:\n")
    # if best is linear there is no gamma parameter
    print(model.best_params_)
    print()
    print("Grid scores on train set:\n")
    means  = model.cv_results_['mean_test_score']
    stds   = model.cv_results_['std_test_score']
    params = model.cv_results_['params']
    for mean, std, params_tuple in zip(means, stds, params):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params_tuple))
    print()
    print("Detailed classification report for the best parameter set:\n")
    print("The model is trained on the full train set.")
    print("The scores are computed on the full test set.\n")
    y_true, y_pred = ytest, model.predict(Xtest)
    print(classification_report(y_true, y_pred))
    print()


# Loop on scores and, for each score, loop on the model labels
"""
iterate varying the score function
    1. iterate varying the classification model among Decision Tree, Naive Bayes, Linear Perceptron, Support Vector
        - activate the grid search
            1. the resulting model will be the best one according to the current score function
        - print the best parameter set and the results for each set of parameters using the above defined function
        - print the classification report
        - store the .best score_ in a dictionary for a final report
    2. print the final report for the current score funtion
"""

results_short = {}

for score in scores:
    print('='*40)
    print("# Tuning hyper-parameters for %s" % score)
    print()

    #'%s_macro' % score ## is a string formatting expression
    # the parameter after % is substituted in the string placeholder %s
    for m in model_lbls:
        print('-'*40)
        print("Trying model {}".format(models[m]['name']))
        clf = GridSearchCV(models[m]['estimator'], models[m]['param'], cv=5,
                           scoring='%s_macro' % score, 
                           iid = False, 
                           return_train_score = False,
                           n_jobs = 2, # this allows using multi-cores
                           )
        clf.fit(Xtrain, ytrain)
        print_results(clf)
        results_short[m] = clf.best_score_
    print("Summary of results for {}".format(score))
    print("Estimator")
    for m in results_short.keys():
        print("{}\t - score: {:4.2}%".format(models[m]['name'], results_short[m]))