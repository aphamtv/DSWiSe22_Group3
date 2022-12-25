#!/usr/bin/env python
# coding: utf-8

# # COMPAS Analysis

# In[1]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[2]:


raw_data = pd.read_csv("./data/compas-scores-two-years.csv")
len(raw_data)


# In[3]:


# filter data
df = raw_data.loc[
    (raw_data['days_b_screening_arrest'] <= 30) &
    (raw_data['days_b_screening_arrest'] >= -30) &
    (raw_data['is_recid'] != -1) &
    (raw_data['c_charge_degree'] != "O") &
    (raw_data['score_text'] != "N/A")
]
len(df)


# In[4]:


# add "length of stay" column = date out - date in
df['length_of_stay'] = pd.to_numeric(pd.to_datetime(df['c_jail_out']) - pd.to_datetime(df['c_jail_in']))


# In[5]:


#calculate the corr btw "length of stay" and "decile score"
print(df['length_of_stay'].corr(df['decile_score']))


# In[6]:


# demographic breakdown
df['age_cat'].value_counts()


# In[7]:


df['race'].value_counts()


# In[ ]:




