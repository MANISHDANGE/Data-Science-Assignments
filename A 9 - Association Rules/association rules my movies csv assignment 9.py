#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[ ]:


#association rules


# In[2]:


data=pd.read_csv("my_movies.csv")
data.head()


# In[3]:


data.describe()


# In[4]:


data.shape


# In[5]:


data.isnull().sum()


# In[6]:


data.dtypes


# In[7]:


data1=data.iloc[:,5:]


# In[8]:


data1


# In[9]:


data1[data1.duplicated()]


# In[11]:


data2=data1.drop_duplicates()
data2


# In[12]:


data2.isnull().sum()


# In[13]:


frequent_itemsets = apriori(data2,min_support = 0.1,use_colnames=True)
frequent_itemsets


# In[15]:


rules = association_rules(frequent_itemsets,metric='lift',min_threshold = 0.7)
rules


# In[16]:


rules1=rules.sort_values('lift',ascending = False)[0:20]


# In[17]:


rules[rules.lift>1]


# In[18]:


rules3 = rules[(rules['lift']>1)&(rules['confidence']>0.7)]
rules3


# In[19]:


freq2_mov=apriori(data2,min_support=0.2,use_colnames=True)
freq2_mov


# In[20]:


rules4=association_rules(freq2_mov,metric='confidence',min_threshold=0.6)
rules4


# In[21]:


rules4[rules4.lift>1]


# In[ ]:


conclusion
Lower the Confidence level Higher the no. of rules.
Higher the Support, lower the no. of rules.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




