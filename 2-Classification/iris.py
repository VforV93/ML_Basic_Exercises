from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline') # the output of plotting commands is displayed inline within frontends like the Jupyter notebook, directly below the code cell that produced it
import numpy   as np
import pandas  as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree    import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#imports for the Decision Tree
from matplotlib import pyplot
from sklearn.tree import plot_tree
from matplotlib.pyplot import figure

# The ﬁle does not have header, use as column names
header = ["sepal lenght","sepal width","petal lenght","petal width","class"]

# Download the Iris dataset at the url
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, sep=',', names=header)

# Show the head of iris dataframe
df.head()

# --- VISUALIZATION ---
# pairplot function of seaborn on the iris dataset
sns.pairplot(df, hue='class', height=3)

# Preparing features and target
# Store in X the content of iris excluding the column species and verify the shape 
X = df.drop(columns=['class'])
np.shape(X) # (150,4)
y = df['class']
np.shape(y) # (150,)


# We will now step through several simple examples of applying supervised and unsupervised learning methods.
# --- Supervised learning example: Iris classiﬁcation ---

# Generate the variables Xtrain, Xtest, ytrain, ytest
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1)
# Show the shape of the resulting variables
np.shape(Xtrain) # (112,4)
np.shape(Xtest)  # (38,4)
np.shape(ytrain) # (112,)
np.shape(ytest)  # (38,)
# Choose the model class DecisionTreeClassifier, instantiate it whithout any hyperparameter and ﬁt the model to data,calling its method fit
dtc = DecisionTreeClassifier()
dtc.fit(Xtrain, ytrain)

ytrain_model = dtc.predict(Xtrain)
print("The accuracy on training set is {}%".format(accuracy_score(ytrain, ytrain_model)*100))

ytest_model = dtc.predict(Xtest)
print("The accuracy on test set is {}%".format(accuracy_score(ytest, ytest_model)*100))


# --- SHOW THE DECISION TREE ---
figure(figsize=[10, 10])
plot_tree(dtc, filled=True, feature_names=header[:-1], 
          class_names=['Iris-setosa','Iris-versicolor','Iris-virginica'],rotate=False)
