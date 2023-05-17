#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import requests


# In[14]:


url = 'https://github.com/MarcelloGalimberti/DMH/blob/main/db_tempi_opt_reale.xlsx?raw=true'


# In[17]:


df_tempi_opt = pd.read_excel(url)


# In[ ]:


# https://github.com/MarcelloGalimberti/DMH/blob/main/db_tempi_opt_reale.xlsx


# In[ ]:





# In[19]:


st.title('MTSV4 MTO Factory Feasibility')
st.write('Database delta tempi bundle')
st.write(df_tempi_opt)


# In[20]:


stgr_installati = st.slider('stgr installati',0,50,(25))
linea_installati = st.slider('linea installati',0,50,(43))


# In[21]:


data = {
    'Area_Produzione': ['stgr', 'linea'],
    'Risorse': [stgr_installati, linea_installati]
}
df = pd.DataFrame(data)


# In[22]:


# Set the Seaborn style and color palette
sns.set_style('dark')
sns.set_palette(sns.color_palette("Reds"))


# In[23]:


fig, ax = plt.subplots()
sns.barplot(x='Area_Produzione', y='Risorse', data=df, ax=ax)
ax.set_xlabel('Area Produzione')
ax.set_ylabel('Risorse')
ax.set_title('Risorse installate')

# Add labels to the bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points', color='white')


# In[24]:


st.pyplot(fig)


# In[ ]:




