
# coding: utf-8

# In[23]:


#First ML Project


# # DECISION TREES (USING SKLEARN DECISIONTREEREGRESSOR)

# In[2]:


#To predict house prices by training a decision tree model using skikit learn Decision Tree Regressor


# In[3]:


#For More Help visit the below link:


# https://www.kaggle.com/dansbecker/

# In[4]:


from sklearn.treeee import DecisionTreeRegressor


# In[5]:


import pandas as pd


# In[7]:


melbourne_data = pd.read_csv('melb_data.csv')


# In[10]:


melbourne_data.head() #This is not important ....I did it just to have a look at dataset and predict on my own


# In[11]:


#Lets say we want to predict prices of first 5 houses....based on the features Rooms,,Bathroom,Landsize,Lattitude,Longtitude


# In[12]:


#First try describing the data to know about it


# In[14]:


melbourne_data.describe()


# In[15]:


melbourne_data.Price.describe()


# In[16]:


#or


# In[17]:


melbourne_data['Price'].describe()


# In[18]:


#Selecting multiple columns at once into a new frame


# In[20]:


melbourne_data[['Price','Longtitude']]


# In[24]:


#Building the predictors set


# In[26]:


predictors_set = melbourne_data[['Rooms','Bathroom','Landsize','Lattitude','Longtitude']]


# In[27]:


predictors_set


# In[28]:


#LEts called it bby a common convention X


# In[29]:


X = predictors_set


# In[31]:


#LEts called price (to be predicted to be y)


# In[32]:


y = melbourne_data['Price']#or melbourne_data.Price


# In[33]:


#Initialize DecisionTreeRegressor into an object variable


# In[34]:


#It would be our model to be trained


# In[35]:


melbourne_model = DecisionTreeRegressor()


# In[36]:


#Now Fit ypur dataset into this


# In[37]:


melbourne_model.fit(X,y)


# In[38]:


#This is the output that describes certain Parameters

