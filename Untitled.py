#!/usr/bin/env python
# coding: utf-8

# ### Type of Forest Cover Prediction

# In this competition you are asked to predict the forest cover type (the predominant kind of tree cover) from strictly cartographic variables (as opposed to remotely sensed data). The actual forest cover type for a given 30 x 30 meter cell was determined from US Forest Service (USFS) Region 2 Resource Information System data. Independent variables were then derived from data obtained from the US Geological Survey and USFS. The data is in raw form (not scaled) and contains binary columns of data for qualitative independent variables such as wilderness areas and soil type.
# 
# This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.

# The study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado
# 
# You are asked to predict an integer classification for the forest cover type. The seven types are:
# 
# 1- Spruce/Fir <br>
# 2- Lodgepole Pine <br>
# 3- Ponderosa Pine <br>
# 4- Cottonwood/Willow <br>
# 5- Aspen <br>
# 6- Douglas-fir <br>
# 7- Krummholz <br>

# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[29]:


train=pd.read_csv("D:\\datas\\Forest\\forest_train.csv")
test=pd.read_csv("D:\\datas\\Forest\\forest_test.csv")


# In[30]:


pd.set_option('display.max_columns',50)


# In[31]:


train.head()


# In[32]:


test.head()


# The training set (15120 observations) contains both features and the Cover_Type. The test set contains only the features. You must predict the Cover_Type for every row in the test set (565892 observations).

# ### Data Fields
# 
# Elevation - Elevation in meters <br>
# Aspect - Aspect in degrees azimuth <br>
# Slope - Slope in degrees <br>
# Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features <br>
# Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features <br>
# Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway <br>
# Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice <br>
# Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice <br>
# Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice <br>
# Horizontal_Distance_To_Fire_Points - Horz Dist to nearest wildfire ignition points <br>
# Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation <br>
# Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation <br>
# Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation <br>

# In[45]:


train.info()


# In[36]:


category=train.columns[-45:]


# In[37]:


train[category]=train[category].astype('category')


# In[34]:


train.isnull().sum()


# no null values

# In[43]:


train=train.drop(['Id'], axis=1)


# In[49]:


train.describe(include='category').T


# In[44]:


train.describe().T


# distance to hyderology and horizontal distance to fire points seems to have some outliers. We will check this this using the box plot

# In[50]:


sns.set(font_scale=1.5)
plt.figure(figsize=(15,20))
n=1
for col in train.select_dtypes('int64'):
    plt.subplot(5,2,n)
    sns.boxplot(train[col])
    n=n+1
    plt.tight_layout()


# vertical distance to hyderology and hillshade 9am has some observation higher than 500 and 0 respectively,but there is a practical possiblility for those observation. 
# 
# other than this there is no significant outliers.

# ### Univariate analysis

# In[55]:


plt.figure(figsize=(15,30))
n=1
for col in train.select_dtypes('int64'):
    plt.subplot(10,2,n)
    sns.distplot(train[col])
    n=n+1
    plt.tight_layout()
plt.show()
    


# most of the observations are left skewed. expect for hill_shade variables. hill shade_3pm shows a gaussian distribution

# In[ ]:





# In[ ]:





# In[ ]:




