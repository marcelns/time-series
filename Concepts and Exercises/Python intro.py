#!/usr/bin/env python
# coding: utf-8

# ## Intro to Time Series

# Forecasting is the process of making predictions based on past and present data. Good forecasts capture the genuine patterns and relationships which exist in the historical data, but do not replicate past events that will not occur again.
# 
# There is a difference between a **random fluctuation** and a **genuine pattern** that should be modelled and extrapolated.

# ### Time series graphics
# The first thing we shoud do in quantitative forecasting is understand the data. This can be done with exploration analysis.
# We should look for:
# 1. Patterns
# 2. Unusual observations
# 2. Changes over time
# 4. Relationships between variables.

# In[2]:


# Pandas is the python library for working with and visualizing time series
import pandas as pd

# Numpy is a library for matricial operations and high-level mathematical functions
import numpy as np

# Matplot lib is the basic python graphic library
import matplotlib.pyplot as plt

# Global parameters for plt graphics
plt.rcParams['font.size'] = 15
plt.rcParams['figure.figsize'] = [10,6]


# In[6]:


# Time series can be thought as a list of numbers indexed by time
# We can define a time series with pandas

t = pd.Series([123,39,78,52,110], index = range(2012,2017))

plt.plot(t);


# In[4]:


help(pd.Series)

