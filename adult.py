from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy   as np
import pandas  as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

header = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation' ,
'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country', 'highincome']
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

df = pd.read_csv("Data/adult.data", sep=',', names=header) # | pd.read_csv(url, sep=',', names=header) 

# Load the data in the dataframe df and then show the column types with the .dtypes attribute of a Pandas DataFrame
df.dtypes

# Show the head and then generate the histograms for all the columns
df.head()
df.hist(figsize = [15,15]) # | pd.DataFrame.hist(df, figsize = [15,15])

# Show a bar graph with the value counts of the attribute high-income.
# Use the method value_counts of Pandas, then plot with the option kind = 'bar'
plt.figure(figsize = [10,10])
df['highincome'].value_counts().plot(kind = 'bar') # | df['highincome'].value_counts().plot.bar()

# Boxplot with Seaborn, hours-per-week in the x axis and high-income in the y axis
plt.figure(figsize = [10,10])
bplot = sns.boxplot(x=df.loc[:,'hours-per-week'], y=df.loc[:,'highincome'], width=0.5, palette='colorblind')

# Something is wrong, the figure does not look like a proper boxplot.
# Let's look at the capital-loss column with the describe method
plt.figure(figsize = [10,10])
sns.boxplot(x=df.loc[:,'capital-loss'], y=df.loc[:,'highincome'])

df['capital-loss'].describe()

# The three quartiles are all zero, and there are no left outliers.
# Let's try with a logarithmic transformation (add +1 to deal with the zero values)
plt.figure(figsize = [10,10])
# Look at the rows with non-zero values: in the x values, instead of the : indicating 'all the rows' we must use a 'selector expression', in this case df['capital-loss']!=0
sns.boxplot(x=np.log10(df.loc[df['capital-loss']!=0,'capital-loss']+1), y=df.loc[:,'highincome'])

# Plot another pair of columns
# education-num and highincome
plt.figure(figsize = [10,10])
sns.boxplot(x=df.loc[:,'education-num'], y=df.loc[:,'highincome'])
