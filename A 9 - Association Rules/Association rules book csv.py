#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlxtend')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[3]:


data = pd.read_csv("book.csv")
data.head()


# In[4]:


data.dtypes


# In[5]:


data.isnull().sum()


# In[6]:


data.corr()


# In[7]:


frequent_itemsets = apriori(data,min_support=0.1,use_colnames=True)
frequent_itemsets


# In[8]:


rules = association_rules(frequent_itemsets,metric='lift',min_threshold = 1)
rules


# In[9]:


rules.sort_values('lift',ascending=False)[0:20]


# In[10]:


rules[rules.lift>1]


# In[14]:


plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[15]:


frequent_book2 = apriori(data,min_support = 0.05,use_colnames = True)
frequent_book2


# In[16]:


rules2 = association_rules(frequent_book2,metric = "confidence",min_threshold = 0.7)
rules2


# In[17]:


rules2[rules2.lift>1]


# In[18]:


plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




