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

# Assuming Tr_X and Tr_y are your training features and labels
Tr_X = df[['age', 'income', 'student', 'credit_rating']]
Tr_y = df['buys_computer']

# Convert categorical variables to numerical
Tr_X = pd.get_dummies(Tr_X)

# Create and fit Gaussian Naive Bayes model
model = GaussianNB()
model.fit(Tr_X, Tr_y)

# Print class conditional densities
print(model.theta_)  # Means of each feature per class
print(model.sigma_)   # Variances of each feature per class


# In[3]:


from scipy.stats import chi2_contingency

# Create a contingency table
contingency_table = pd.crosstab(df['buys_computer'], [df['age'], df['income'], df['student'], df['credit_rating']])

# Perform chi-square test for independence
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-square value: {chi2}")
print(f"P-value: {p}")


# In[4]:


# Assuming X_test is your test data
X_test = pd.DataFrame({'age': ['<=30'], 'income': ['medium'], 'student': ['yes'], 'credit_rating': ['fair']})

# Convert categorical variables to numerical
X_test = pd.get_dummies(X_test)

# Use the trained model to make predictions
predictions = model.predict(X_test)
print(predictions)


# In[ ]:




