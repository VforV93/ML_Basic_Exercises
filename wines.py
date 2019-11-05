import numpy as np
import pandas as pd

# Read data from file 'filename.csv' 
df = pd.read_csv("Data/winequality-red.csv", sep=';')

# Show column names 
df.columns

# Show portion of data
df.head()

# Show histograms for all numeric values
hist = df.hist()

# Show synthetic description
sd = df.describe()
sd["quality"]


