#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sdv.tabular import CTGAN


# In[6]:


import pandas as pd


# In[2]:


model = CTGAN.load("AssistmentsGAN.pkl")


# In[15]:


real_data = pd.DataFrame(pd.read_csv("df_sorted_1_2.csv"))


# In[9]:


real_data.shape


# In[18]:


synthetic_data = pd.DataFrame(model.sample(num_rows=10**5))


# ## Statistical Metrics

# In[3]:


from sdv.metrics.tabular import CSTest, KSTest


# In[19]:


KSTest.compute(real_data, synthetic_data)


# ## Detection Metrics

# In[22]:


from sdv.metrics.tabular import LogisticDetection


# In[23]:


LogisticDetection.compute(real_data, synthetic_data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




