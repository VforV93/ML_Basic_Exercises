# ML_Basic_Exercises
Some Machine Learning basic execises

DATA EXPLORATION
1) Wines Quality Dataset
    - Read data from archive
        In this case, it is a csv with header, separator is ‘;’ The download url is http://archive.ics.uci.edu/ml/machine-learning-databases/winequality/winequality-red.csv
        Use the read_csv() method of pandas dataframe https://pandas.pydata.org/pandasdocs/stable/reference/api/pandas.read_csv.html 
        Use df as the dataframe name 
        In this dataset the column names are already included in the .csv ﬁle

2) Iris Dataset
    - Look at the ".names" text ﬁle in the Data Folder, read (visually) the  column names and store them in a list 
    - The url is ’https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data’ 
    - Read the ﬁle with read_csv using also the names parameter
    - Show the head of the ﬁle, just for a quick inspection

3) Adult Dataset
    - Look at the ".names" text ﬁle in the Data Folder, read (visually) the  column names and store them in a list 
    - url = ‘https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data’ 
    - Print all the types of the columns using the types attribute
    - Load the data in the dataframe df and then show the column types with the .dtypes attribute of a Pandas DataFrame
    - Show the head and then generate the histograms for all the columns

CLASSIFICATION WITH DECISION TREES
1) Introducint Scikit-Learn - intro Iris -
    - Data Repesentation in Scikit-Learn
        -> Data as table: a two-dimensional grid of data
                         rows    - individualk elements
                         columns - quantities related to each elements
    
    We may wish to construct a model that can predict the species of ﬂower based on the other measurements. The measurements of the ﬂower components are the features array and the species column can be considered the target array.

    - Scikit-Learn’s Estimator API 
        TheScikit-LearnAPIisdesignedwiththefollowingguidingprinciplesinmind:
        - Consistency: Allobjectsshareacommoninterfacedrawnfromalimitedsetofmethods,with consistent documentation.
        - Inspection: All speciﬁed parameter values are exposed as public attributes. 
        - Limited object hierarchy: Only algorithms are represented by Python classes; datasets are represented in standard formats (NumPy arrays, Pandas DataFrames, SciPy sparse matrices) and parameter names use standard Python strings.
        -  Composition: Many machine learning tasks can be expressed as sequences of more fundamental algorithms, and Scikit-Learn makes use of this wherever possible.
        -  Sensible defaults: When models require user-speciﬁed parameters, the library deﬁnes an appropriate default value.
        In practice, these principles make Scikit-Learn very easy to use, once the basic principles are understood.
    
    - Basics of the API
        The steps in using the Scikit-Learn estimator API are as follows 
        - Choose a class of model by importing the appropriate estimator class from Scikit-Learn.
        - Choose model hyperparameters by instantiating this class with desired values.
        - Arrange data into a features matrix and target vector.
        - Fit the model to your data by calling the fit() method of the model instance.
        - Apply the Model to new data

2) Pruning the Decision Tree
    In this example we are directly given two different datasets, one will be used for training, the other for testing. We will start training the model with the training data, then testing it with the test data. Then we will observe the resulting tree, and try to improve the result with pruning.
        -  Train the full tree
        -  Observe the tree
            Try to understand better the tree by plotting only the first two levels under the root. This is obtained with the parameter max_depth = 3. Remember that here we are not changing the tree, but only displaying the upper part of the tree.
        -  Pruned tree
            From the observation of the tree, choose an appropriate value for max_dept and redo the training using the parameter max_depth = max_depth in the fit method. Compute the accuracy on the training set, and then on the test set.

            The accuracy of the pruned tree on training set is 75.0%
            The accuracy of the pruned tree on test set is 76.6%

CLASSIFICATION WITH HYPERPARAMETER TUNING

aim:

Show classification with different strategies for the tuning and evaluation of the classifier 1. simple holdout 2. holdout with validation train and validate repeatedly changing a hyperparameter, to find the value giving the best score, then test for the final score 3. cross validation on training set, then score on test set 4. bagging it is an ensemble method made available in scikit-learn

NB: You should not interpret those experiments as a way to find the best evaluation method, but simply as examples of how to do the evaluation.

If you look at the final report, methods 1 to 3 are meant for increasing evaluation reliability, method 3 is the more reliable, but it requires several repetitions for cross validation, therefore, if the learning method is expensive, it requires long processing time. If, due to intrinsic variation caused by random sampling, it turns out that methods 1 or 2 give higher accuracy, this means simply that the forecast towards generalisation is less reliable.

Method 4 is on a different dimension, it simply shows that a good result can be obtained with an ensemble of simpler classifiers (the best value for the hyperparameter max_depth is smaller than in the other cases)

Workflow
    -  download the data
    -  drop the useless data
    -  separe the predicting attributes X from the class attribute y
    -  split X and y into training and test

1) part 1 - single run with default parameters
    -  initialise an estimator with the chosen model generator
    -  fit the estimator with the training part of X
    -  show the tree structure
    part 1.1
        -  predict the y values with the fitted estimator and the train data
            -  compare the predicted values with the true ones and compute the accuracy on the training set
    part 1.2
        -  predict the y values with the fitted estimator and the test data
            -  compare the predicted values with the true ones and compute the accuracy on the test set

2) part 2 - multiple runs changing a parameter
    -  prepare the structure to hold the accuracy data for the multiple runs
    -  repeat for all the values of the parameter
        -  initialise an estimator with the current parameter value
        -  fit the estimator with the training part
        -  predict the class for the test part
        -  compute the accuracy and store the value
    -  find the parameter value for the top accuracy

3) part 3 - compute accuracy with cross validation
    -  prepare the structure to hold the accuracy data for the multiple runs
    -  repeat for all the values of the parameter
        -  initialise an estimator with the current parameter value
        -  compute the accuracy with cross validation and store the value
    -  find the parameter value for the top accuracy
    -  fit the estimator with the entire X
    -  show the resulting tree and classification report

4) part 4 - compute accuracy with bagging validation
    -  prepare the structure to hold the accuracy data for the multiple runs
    -  repeat for all the values of the parameter
        -  initialise an estimator with the current parameter value
        -  compute the accuracy with bagging validation and store the value
    -  find the parameter value for the top accuracy
    -  fit the estimator with the entire X
    -  show the resulting tree and classification report
