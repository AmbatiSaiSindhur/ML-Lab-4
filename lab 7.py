#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#for dataset 1


# In[19]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
 
# Loading the data
data = pd.read_csv("C:\class\projects\sem 5\Machine Learning\Custom_CNN_Features.csv")
 
features = data[['f8', 'f10']]
target = data['Label']
 
# Spliting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
 
# Initializing 
clf = SVC()
clf.fit(X_train, y_train)
 
support_vectors = clf.support_vectors_
 
# Printing the vectors
print(f'Supporting Vectors ={support_vectors}')
 


# In[20]:


#A2
 
# Testing the accuracy 
accuracy = clf.score(X_test[['f8', 'f10']], y_test)
print(f"Accuracy of the SVM : {accuracy}")
 
# Classification for the given test vector
test_vector = X_test[['f8', 'f10']].iloc[0]
predicted_class = clf.predict([test_vector])
print(f"The predicted class for the test vector: {predicted_class}")


# In[21]:


features = data[['f8', 'f10']]
target = data['Label']
 
# Spliting the data 
X_train, X_test, y_atrain, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# Making predictions 
predictions = clf.predict(X_test)


print("Predictions:", predictions)

for i, prediction in enumerate(predictions):
    print(f"Sample {i + 1}: Predicted class {prediction}")

accuracy = sum(predictions == y_test) / len(y_test)
print("Accuracy:", accuracy)


# In[22]:


from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming 'X_train', 'X_test', 'y_train', 'y_test' are your feature and target variables

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Iterate over different kernel functions
kernel_functions = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernel_functions:
    # Create SVM model with the specified kernel
    model = svm.SVC(kernel=kernel)
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print the results
    print(f'Kernel: {kernel}, Accuracy: {accuracy}')


# In[ ]:


#for dataset2


# In[23]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
 
# Loading the dataset
data = pd.read_csv("C:\class\projects\sem 5\Machine Learning\palm_document_Gabor.csv")
 
features = data[['Theta1_Lambda1_MeanAmplitude', 'Theta1_Lambda1_LocalEnergy']]
target = data['Label']
 
# Spliting the data 
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
 
# Initializing the Support Vector Machine (SVM) 
clf = SVC()
clf.fit(X_train, y_train)
 
support_vectors = clf.support_vectors_
 
# Printing the support vectors
print(f'Support Vectors ={support_vectors}')
 


# In[24]:


#A2
 
# Testing the accuracy 
accuracy = clf.score(X_test[['Theta1_Lambda1_MeanAmplitude', 'Theta1_Lambda1_LocalEnergy']], y_test)
print(f"Accuracy of the SVM on the test set: {accuracy}")
 
# Performing classification 
test_vector = X_test[['Theta1_Lambda1_MeanAmplitude', 'Theta1_Lambda1_LocalEnergy']].iloc[0]
predicted_class = clf.predict([test_vector])
print(f"The predicted class for the test vector: {predicted_class}")


# In[25]:


features = data[['Theta1_Lambda1_MeanAmplitude', 'Theta1_Lambda1_LocalEnergy']]
target = data['Label']
 
# Spliting the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# Making predictions 
predictions = clf.predict(X_test)

print("Predictions:", predictions)

for i, prediction in enumerate(predictions):
    print(f"Sample {i + 1}: Predicted class {prediction}")

accuracy = sum(predictions == y_test) / len(y_test)
print("Accuracy:", accuracy)


# In[26]:


from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Iterate over different kernel functions
kernel_functions = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernel_functions:
    model = svm.SVC(kernel=kernel)
    
    # Training the model
    model.fit(X_train_scaled, y_train)
    
    # Makeing predictions 
    y_pred = model.predict(X_test_scaled)
    
    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Printing the results
    print(f'Kernel: {kernel}, Accuracy: {accuracy}')

