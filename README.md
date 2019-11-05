# ML_Basic_Exercises
Some Machine Learning basic execises


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
