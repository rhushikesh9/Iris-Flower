#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load libraries
from pandas import read_csv
# from pandas.tools.plotting import scatter_matrix (https://github.com/pandas-dev/pandas/issues/15893)
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


import os


# In[6]:


dataset = read_csv(r"C:\Users\rkadu\OneDrive\Documents\iris.CSV")


# In[7]:


dataset


# In[8]:


# df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)
dataset.rename(columns={'SepalLengthCm': 'sepal-length', 'SepalWidthCm': 'sepal-width', 'PetalLengthCm':'petal-length', 'PetalWidthCm':'petal-width', 'Species':'class'}, inplace=True)

# drop the Id column
dataset = dataset.drop('Id', 1)


# In[11]:


dataset


# In[ ]:


##Summarize the Dataset
#In this step we are going to take a look at the data in a few different ways:

#Dimensions of the dataset.
#Peek at the data itself.
#Statistical summary of all attributes.
#Breakdown of the data by the class variable.


# In[12]:


##Dimensions of Dataset
#We can get a quick idea of the number of instances (rows) and number of attributes (columns).


# In[13]:


# shape
print(dataset.shape)


# In[16]:


##Peek at the Data
#Let's eyeball our data.


# In[17]:


print(dataset.head(5))


# In[ ]:


#Statistical Summary
#Let's take a look at a summary of each attribute. This includes the count, mean, the min and max values as well as some percentiles


# In[18]:


print(dataset.describe())


# In[ ]:


#Class Distribution
#Let's now take a look at the number of instances (rows) that belong to each class. We can view this as an absolute count.

#On classification problems we need to know how balanced the class values are.
#Highly imbalanced problems (a lot more observations for one class than another) are common and may need special handling in the data preparation stage of our project.


# In[19]:


# class distribution
print(dataset.groupby('class').size())


# In[ ]:


##Data Visualization
#We now have a basic idea about the data. We need to extend this with some visualizations. We are going to look at two types of plots:

#Univariate plots to better understand each attribute.
#Multivariate plots to better understand the relationships between attributes


# In[ ]:


#Univariate Plots
#We will start with some univariate plots, that is, plots of each individual variable. Given that the input variables are numeric, we can create box and whisker plots of each.


# In[20]:


# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()


# In[21]:


# histograms
dataset.hist()
pyplot.show()


# In[ ]:


#Multivariate Plots
#Now we can look at the interactions between the variables. Let's look at scatter plots of all pairs of attributes. This can be helpful to spot structured relationships between input variables.


# In[22]:


# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()


# In[ ]:


#Evaluate Some Algorithms
#Now it is time to create some models of the data and estimate their accuracy on unseen data. Here is what we are going to cover in this step:

#Separate out a validation dataset.
#Setup the test harness to use 10-fold cross validation.
#Build 6 different models to predict species from flower measurements
#Select the best model.


# In[ ]:


#reate a Validation Dataset
#We need to know whether or not the model that we created is any good. Later, we will use statistical methods to estimate the accuracy of the models that we create on unseen data. We also want a more concrete estimate of the accuracy of the best model on unseen data by evaluating it on actual unseen data. That is, we are going to hold back some data that the algorithms will not get to see and we will use this data to get a second and independent idea of how accurate the best model might actually be. We will split the loaded dataset into two, 80% of which we will use to train our models and 20% that we will hold back as a validation dataset.


# In[23]:


# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[ ]:


#Test Harness
#We will use 10-fold cross validation to estimate accuracy. This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits. We are using the metric of accuracy to evaluate models. This is a ratio of the number of correctly predicted instances divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). We will be using the scoring variable when we run build and evaluate each model next.


# In[ ]:


#Build Models
#We don't know which algorithms would be good on this problem or what configurations to use. We get an idea from the plots that some of the classes are partially linearly separable in some dimensions, so we are expecting generally good results. Let's evaluate six different algorithms:


# In[ ]:


#Logistic Regression (LR)
#Linear Discriminant Analysis (LDA)
#k-Nearest Neighbors (KNN)
#Classifcation and Regression Trees (CART)
#Gaussian Naive Bayes (NB)
#Support Vector Machines (SVM)


# In[ ]:


#This list is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms. We reset the random number seed before each run to ensure that the evaluation of each algorithm is performed using exactly the same data splits. It ensures the results are directly comparable. Let's build and evaluate our six models:

#We now have training data in the X_train and Y_train for preparing models and a X_validation and Y_validation sets that we can use later.


# In[25]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# In[27]:


results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=None)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[28]:


# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[ ]:


##Predictions
#Out of LR, KNN and SVM, we don't know which will be most accurate on our validation dataset. Let's find out. We will run these models directly on the validation set and summarize the results as a final accuracy score, a confusion matrix and a classification report.
#This will give us an independent final check on the accuracy of the best model. It is important to keep a validation set just in case we made a slip during training, such as overfitting to the training set or a data leak. Both will result in an overly optimistic result.


# In[ ]:


#predictions on validation dataset based on LogisticRegression


# In[29]:


lr = LogisticRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[ ]:


#Predictions on validation dataset based on KNeighborsClassifier


# In[30]:


# predictions on validation dataset based on KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[ ]:


#Predictions on validation dataset based on SVC


# In[31]:


#predictions on validation dataset based on SVC
svc = SVC()
svc.fit(X_train, Y_train)
predictions = svc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[ ]:




