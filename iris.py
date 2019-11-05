import numpy  as np
import pandas as pd

header = ["sepal lenght","sepal width","petal lenght","petal width","class"]

df = pd.read_csv("Data/iris.data", sep=',', names=header)

# Print histogram of numeric values
hist = df.hist()

# Print histogram of frequencies for the class value
sd = df.describe()
class_series = df['class'].value_counts()
class_series.plot.bar()