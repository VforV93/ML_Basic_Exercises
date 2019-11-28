import warnings
warnings.filterwarnings('ignore') # uncomment this line to suppress warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

print(__doc__) # print information included in the triple quotes at the beginning

random_state = 42


# url = 'diagnosis.data'
# names = ['Temp', 'Nau', 'Lum', 'Uri', 'Mic', 'Bur', 'd1', 'd2']
# sep = "\t"
url = 'Data/breast-cancer.data'
names = ['Class','age','menopause','tumor-size','inv-nodes',
         'node-caps','deg-malig','breast','breast-quad','irradiat']
sep = ","

df = pd.read_csv(url, names = names, sep=sep)
print(df.head())

#Show the types of the columns
print("\nTypes of the columns")
print(df.dtypes)

# -- Clean the column tumor-size
tumor_size_dict = dict(zip(list(df['tumor-size'].unique()),list(df['tumor-size'].unique())))
tumor_size_dict['0-4'] = '00-04'
tumor_size_dict['5-9'] = '05-09'
#print(tumor_size_dict)
df['tumor-size'] = df['tumor-size'].map(tumor_size_dict)
df['tumor-size']


# -- Clean the column inv-nodes
inv_nodes_dict = dict(zip(list(df['inv-nodes'].unique()),list(df['inv-nodes'].unique())))
inv_nodes_dict['0-2']  = '00-02'
inv_nodes_dict['6-8']  = '06-08' 
inv_nodes_dict['9-11'] = '09-11'
inv_nodes_dict['3-5']  = '03-05'
#print(inv_nodes_dict)
df['inv-nodes'] = df['inv-nodes'].map(inv_nodes_dict)
df['inv-nodes']

# Inspect the data again
print(df.head())


#Prepare the lists of numeric features, ordinal features, categorical features
non_numeric_features = ['Class','age','menopause','tumor-size','inv-nodes','node-caps','breast','breast-quad','irradiat']
numeric_features     = ['deg-malig']
ordinal_features     = ['age', 'tumor-size', 'inv-nodes']
categorical_features = ['menopause', 'irradiat', 'breast', 'node-caps', 'breast-quad']


# -- Prepare the transformer
# transf_dtype = np.float64
transf_dtype = np.int32

categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse = False, dtype = transf_dtype)
ordinal_transformer     = OrdinalEncoder(dtype = transf_dtype)
preprocessor = ColumnTransformer(
    transformers = [('cat', categorical_transformer, categorical_features),
                    ('ord', ordinal_transformer, ordinal_features)
                   ],
                    remainder = 'passthrough'
    )

# Preparing features and target
target_name = 'Class'
X = df.drop(columns=[ target_name ]) # drop the Class column
print(np.shape(X)) # (286,9)
y = df[ target_name ] # Class only
print(np.shape(y)) # (286,)

preprocessor.fit(X, y)

print(preprocessor.named_transformers_)
X_p = preprocessor.fit_transform(X)
print(np.shape(X_p))
print(X_p[0:5,:])
df_p = pd.DataFrame(
    data = X_p
)
print(df_p.head())

df_p.describe()
# Generate the variables Xtrain, Xtest, ytrain, ytest
#Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=random_state) # default Train 0.75- Test 0.25