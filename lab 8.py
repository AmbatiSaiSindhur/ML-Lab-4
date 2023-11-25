#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Creating a DataFrame with the provided data
data = {
    'age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(data)

# Calculate prior probabilities
total_instances = len(df)
class_counts = df['buys_computer'].value_counts()

prior_probabilities = class_counts / total_instances

# Display the prior probabilities
print("Prior Probabilities:")
print(prior_probabilities)


# In[2]:


from sklearn.naive_bayes import GaussianNB
import numpy as np

# Convert categorical features to numerical values
df_numeric = pd.get_dummies(df[['age', 'income', 'student', 'credit_rating']])

# Split data into features and target
X = df_numeric.values
y = df['buys_computer'].values

# Fit a Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X, y)

# Get the class conditional densities
class_conditional_densities = np.exp(model.theta_)
print("Class Conditional Densities:")
print(pd.DataFrame(class_conditional_densities, columns=df_numeric.columns))


# In[3]:


from scipy.stats import chi2_contingency

# Create a contingency table
contingency_table = pd.crosstab(df['buys_computer'], [df['age'], df['income'], df['student'], df['credit_rating']])

# Perform chi-square test for independence
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-square value: {chi2}")
print(f"P-value: {p}")


# In[4]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Convert categorical features to numerical values
df_numeric = pd.get_dummies(df[['age', 'income', 'student', 'credit_rating']])

# Split data into features and target
X = df_numeric.values
y = df['buys_computer'].values

# Split data into training and testing sets
Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the Naïve-Bayes classifier
model = GaussianNB()
model.fit(Tr_X, Tr_y)

# Make predictions on the test set
predictions = model.predict(Te_X)

# Evaluate accuracy
accuracy = accuracy_score(Te_y, predictions)
print("Accuracy:", accuracy)




# In[5]:
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your data from the CSV file
df_numeric = pd.read_csv(r"C:\class\projects\sem 5\Machine Learning\Custom_CNN_Features.csv")

# Assuming 'f8' is the feature and 'f0' is the target
features = df_numeric[['f8','f10']]
target = df_numeric['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Build and train the Naïve-Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)




