#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
df=pd.read_csv("C:\class\projects\sem 5\Machine Learning\Custom_CNN_Features.csv")
dataframe=df.drop(columns=['Filename'])
print(dataframe)


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

#Loading the  dataset
df=pd.read_csv("C:\class\projects\sem 5\Machine Learning\Custom_CNN_Features.csv")

#Loading the features with numeric values
feature_a = df['f8']
feature_b = df['f190']

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(feature_a, feature_b, color='blue', alpha=0.5)

# Adding labels
plt.xlabel('f8')
plt.ylabel('f190')
plt.title('Scatter Plot of f8 vs f190')

# Display plot
plt.grid(True)
plt.show()


# In[3]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# Loading the dataset
df=pd.read_csv("C:\class\projects\sem 5\Machine Learning\Custom_CNN_Features.csv")

# Assumeing the independent and dependent number
independent_number = df['f190']
dependent_number = df['f8']

# Reshape the data as sklearn's LinearRegression model expects 2D array
independent_number = independent_number.values.reshape(-1, 1)
dependent_number = dependent_number.values.reshape(-1, 1)

# Linear regression model
model = LinearRegression()
model.fit(independent_number, dependent_number)

# Predict the values
predicted_values = model.predict(independent_number)

# Mean squared error
mse = mean_squared_error(dependent_number, predicted_values)
print(f'Mean Squared Error: {mse}')

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(independent_number, dependent_number, color='blue', alpha=0.5)

# Regression line
plt.plot(independent_number, predicted_values, color='red')

# Adding the labels
plt.xlabel('f8')
plt.ylabel('f190')
plt.title('Linear Regression Model: f8 vs f190')

# Display plot
plt.grid(True)
plt.show()


# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Loading dataset
df=pd.read_csv("C:\class\projects\sem 5\Machine Learning\Custom_CNN_Features.csv")


# Initialize the Logistic Regression model
logistic_model = LogisticRegression()

binary_dataframe = df[df['Label'].isin([0, 1])]
X = binary_dataframe[['f190', 'f8']]
y = binary_dataframe['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Logistic Regression model on the training data
logistic_model.fit(X_train, y_train)

# Making predictions on the test set
predictions = logistic_model.predict(X_test)

# Calculate accuracy 
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of Logistic Regression on the test set: {accuracy * 100:.2f}%")


# In[5]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Loading dataset
df=pd.read_csv("C:\class\projects\sem 5\Machine Learning\Custom_CNN_Features.csv")

# Assuming your target number is 'target_number'
target_number = df['Label']

# Extracting features
X = df[['f8', 'f190']]

# Spliting the data 
X_train, X_test, y_train, y_test = train_test_split(X, target_number, test_size=0.2, random_state=42)

# Mean squared error Decision Tree Regressor
reg_tree = DecisionTreeRegressor(random_state=42)
reg_tree.fit(X_train, y_train)
y_pred_tree = reg_tree.predict(X_test)
meansqerror_tree = mean_squared_error(y_test, y_pred_tree)
print(f"Decision Tree Mean Squared Error: {meansqerror_tree}")

# Mean squared error k-NN Regressor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train_scaled, y_train)
y_pred_knn = knn_regressor.predict(X_test_scaled)
meansqerror_knn = mean_squared_error(y_test, y_pred_knn)
print(f"k-NN Regressor Mean Squared Error: {meansqerror_knn}")


# In[ ]:


#Dataset no:2


# In[7]:


import pandas as pd
data1=pd.read_csv("C:\class\projects\sem 5\Machine Learning\palm_document_Gabor.csv")
print(data1.head())


# In[8]:


import pandas as pd

# Loading the data
df = pd.read_csv("C:\class\projects\sem 5\Machine Learning\palm_document_Gabor.csv")

# Define a function to change values in the "Label" column
def change_label(row):
    if row['Label'] == 'bad':
        return 2
    if row['Label']== 'good':
        return 0
    if row['Label']== 'medium':
        return 1

# Apply the function to the "Label" column
df['Label'] = df.apply(change_label, axis=1)

# Loading the modified DataFrame to an Excel file
df.to_csv('modified_dataset.csv', index=False)
print(df.head(15))


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt

#Load the dataset
df = pd.read_csv("C:\class\projects\sem 5\Machine Learning\palm_document_Gabor.csv")

#Loading 2 features having numeric values
feature_1 = df['Theta1_Lambda0_LocalEnergy']
feature_2 = df['Theta1_Lambda0_MeanAmplitude']

# Scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(feature_1, feature_2, color='blue', alpha=0.5)

# Adding labels
plt.xlabel('Theta0_Lambda1_LocalEnergy')
plt.ylabel('Theta0_Lambda1_MeanAmplitude')
plt.title('Scatter Plot of feature 1 vs feature 2')

# Display plot
plt.grid(True)
plt.show()


# In[10]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("C:\class\projects\sem 5\Machine Learning\palm_document_Gabor.csv")

# Assuming the independent and dependent number
independent_number = df['Theta1_Lambda0_MeanAmplitude']
dependent_number = df['Theta1_Lambda0_LocalEnergy']

# Reshape the data as sklearn's LinearRegression model expects 2D array
independent_number = independent_number.values.reshape(-1, 1)
dependent_number = dependent_number.values.reshape(-1, 1)

# Create a linear regression model
model = LinearRegression()
model.fit(independent_number, dependent_number)

# Predicting values
predicted_values = model.predict(independent_number)

# mean squared error
mse = mean_squared_error(dependent_number, predicted_values)
print(f'Mean Squared Error: {mse}')

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(independent_number, dependent_number, color='blue', alpha=0.5)

# Ploting regression line
plt.plot(independent_number, predicted_values, color='red')

# Add labels and title
plt.xlabel('Theta1_Lambda0_LocalEnergy')
plt.ylabel('Theta1_Lambda0_MeanAmplitude')
plt.title('Linear Regression Model: feature1 vs feature2')

# Display plot
plt.grid(True)
plt.show()


# In[11]:


import pandas as pd

# Load the Excel file into a DataFrame
df = pd.read_csv("C:\class\projects\sem 5\Machine Learning\palm_document_Gabor.csv")

# Define a function to change values in the "Label" column
def change_label(row):
    if row['Label'] == 'bad':
        return 2
    if row['Label']== 'good':
        return 0
    if row['Label']== 'medium':
        return 1

# Apply the function to the "Label" column
df['Label'] = df.apply(change_label, axis=1)

# Save the modified DataFrame back to an Excel file
df.to_csv('modified_dataset.csv', index=False)
print(df.head(10))


# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Loading dataset
# Initialize the Logistic Regression model
logistic_model = LogisticRegression()
#initialize X,y
binary_dataframe = df[df['Label'].isin([0, 1])]
X = binary_dataframe[['Theta1_Lambda0_MeanAmplitude', 'Theta1_Lambda0_LocalEnergy']]
y = binary_dataframe['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Logistic Regression model on the training data
logistic_model.fit(X_train, y_train)

# Making predictions on the test set
predictions = logistic_model.predict(X_test)

# Calculating accuracy 
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of Logistic Regression on the test set: {accuracy * 100:.2f}%")


# In[13]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
# Assuming your target variable is 'target_number'
target_number = df['Label']

# Extracting features
X = df[['Theta1_Lambda0_LocalEnergy','Theta1_Lambda0_MeanAmplitude']]

# Spliting the data 
X_train, X_test, y_train, y_test = train_test_split(X, target_number, test_size=0.2, random_state=42)

# Mean squared error Decision Tree Regressor
reg_tree = DecisionTreeRegressor(random_state=42)
reg_tree.fit(X_train, y_train)
y_pred_tree = reg_tree.predict(X_test)
mean_sqerror_tree = mean_squared_error(y_test, y_pred_tree)
print(f"Decision Tree Mean Squared Error: {mean_sqerror_tree}")

# Mean squared error k-NN Regressor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train_scaled, y_train)
y_pred_knn = knn_regressor.predict(X_test_scaled)
mean_sqerror_knn = mean_squared_error(y_test, y_pred_knn)
print(f"k-NN Regressor Mean Squared Error: {mean_sqerror_knn}")

