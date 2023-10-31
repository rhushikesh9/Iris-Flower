#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##Unemployment Analysis


# In[1]:



import numpy as np
import pandas as pd 


# In[3]:


import os


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[5]:


df=pd.read_csv(r"C:\Users\rkadu\Downloads\unemployment.csv")
df


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.dropna(inplace=True)
df


# In[9]:


df.columns


# In[ ]:


##Formatting column names


# In[10]:


#removing trailing and leading spaces from column names 
df.columns = df.columns.str.strip()


# In[11]:


df['Frequency'].unique()


# In[12]:


df["Frequency"]=df["Frequency"].str.strip()


# In[13]:


df['Frequency'].unique()


# In[16]:


# Converting to datetime format
df['Date'] = pd.to_datetime(df['Date'],format=None)


# In[17]:


df


# In[ ]:


#Extracting Month


# In[18]:


import calendar 
df['month_int'] = df['Date'].dt.month
df['month'] = df['month_int'].apply(lambda x: calendar.month_abbr[x])
df.head()


# In[ ]:


#Month-Wise Visualization


# In[19]:


sns.histplot(data=df, x="Estimated Unemployment Rate (%)", hue="month", multiple="stack")
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
plt.legend(title="Month", loc="upper right", labels=month_order)
# multiple="stack", the bars for different categories (months) are stacked on top of each other.


# In[20]:


sns.histplot(data=df, x="Estimated Employed", hue="month", multiple="stack")
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
plt.legend(title="Month", loc="upper right", labels=month_order)


# In[21]:


data = df.groupby(['month'])[['Estimated Unemployment Rate (%)','Estimated Employed','Estimated Labour Participation Rate (%)']].mean()
data=pd.DataFrame(data).reset_index()


# In[22]:


data


# In[23]:


sns.barplot(x="Estimated Labour Participation Rate (%)", y="month",data=data)


# In[24]:


sns.barplot(x="Estimated Employed", y="month",data=data)


# In[ ]:


#State-Wise Analysis


# In[25]:


## Grouping state wise
state =  df.groupby(['Region'])[['Estimated Unemployment Rate (%)','Estimated Employed','Estimated Labour Participation Rate (%)']].mean()
state = pd.DataFrame(state).reset_index()


# In[26]:


fig = px.box(data_frame=df,x='Region',y='Estimated Unemployment Rate (%)',color='Region',title='Unemployment rate')
fig.update_layout(xaxis={'categoryorder':'total descending'})


# In[27]:


fig = px.bar(state,x='Region',y='Estimated Labour Participation Rate (%)',color='Region',title='Average Labour Participation Rate (State)')
fig.update_layout(xaxis={'categoryorder':'total descending'})


# In[ ]:


#Area-Wise Analysis


# In[28]:


df.Area.unique()


# In[29]:


Area_data = df.groupby(['Area'])[['Estimated Unemployment Rate (%)','Estimated Employed','Estimated Labour Participation Rate (%)']].mean()
Area_data = pd.DataFrame(Area_data).reset_index()


# In[30]:


px.scatter_matrix(df,dimensions=['Estimated Unemployment Rate (%)','Estimated Employed','Estimated Labour Participation Rate (%)'],color='Area',
                 width=900, height=800)


# In[31]:


fig = px.bar(Area_data,x='Area',y='Estimated Labour Participation Rate (%)',color='Area',title='Estimated Unemployment Rate (%)',
            width=400,height=400)
fig.update_layout(xaxis={'categoryorder':'total descending'})


# In[ ]:


#Area-State Wise


# In[32]:


region_area_data = df.groupby(['Area', 'Region'])[['Estimated Labour Participation Rate (%)', 'Estimated Employed']].mean().reset_index()
region_area_data.head()


# In[33]:


px.sunburst(region_area_data,path=['Area','Region'],values='Estimated Employed',
                 title ='Employment rate in State and Area',height=500,width=500)


# In[ ]:




