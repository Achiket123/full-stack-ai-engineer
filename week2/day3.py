import pandas as pd

# pandas data structures

# Series 1-D labeled array holding any data type
s = pd.Series([1,2,3,4,5],index=['a','b','c','d','e'])
print(s)
# Dataframe 2-D labeled data structure with columns of potentially different types like table
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 35, 40, 44],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']}
df = pd.DataFrame(data)
print(df)

# Loading data from csv
data = pd.read_csv('data-2.csv')
print(data.head()) # print first 5 rows of the dataframe
print(data.tail(3))
data.info() # summary of the dataframe including number of non-null entries and data types

print(data.describe()) 


print(data['HadCRUT5'].mean()) # mean HadCRUT5
print(data['HadCRUT5'].median()) # median HadCRUT5
print(data['HadCRUT5'][data['HadCRUT5'] > 0].std()) # standard deviation of HadCRUT5
print(data['HadCRUT5'].var()) # variance of HadCRUT5
print(data['HadCRUT5'].min()) # minimum HadCRUT5

print(data.iloc[0,5]) # first row of the dataframe
print(data.iloc[:,0]) # first 5 rows of the dataframe

# IRIS dataset handson

df = pd.read_csv('iris.csv')
# explore the structure
print("First 5 rows")
df.info()
print(df.describe())
head = df.head()
print(head)
print("Last 5 rows")
tail = df.tail(5)
print(tail)

# select specifc columns and filter rows sepal length >5

selected_col = df[["species" ,"sepal_length"]] 
print(selected_col)

filtered_rows = df[(df["sepal_length"] > 5) & (df["species"] == "setosa")]
print(filtered_rows)
