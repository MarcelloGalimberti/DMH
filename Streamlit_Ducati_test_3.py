#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import requests
import plotly.express as px


# #### Importo tabelle da GitHub

# In[2]:


url_tempi_opt = 'https://github.com/MarcelloGalimberti/DMH/blob/main/db_tempi_opt_reale.xlsx?raw=true'


# In[3]:


df_tempi_opt = pd.read_excel(url_tempi_opt,index_col = 'Bundle')


# In[5]:


url_take_rate = 'https://github.com/MarcelloGalimberti/DMH/blob/main/db_take_rate.xlsx?raw=true'


# In[3]:


df_take_rate = pd.read_excel(url_take_rate)


# In[8]:


url_vincoli = 'https://github.com/MarcelloGalimberti/DMH/blob/main/Vincoli_produzione.xlsx?raw=true'


# In[9]:


vincoli_produzione = pd.read_excel(url_vincoli,index_col='Origine')


# In[11]:


url_tempi_base = 'https://github.com/MarcelloGalimberti/DMH/blob/main/db_tempi_base.xlsx?raw=true'


# In[12]:


tempi_base = pd.read_excel(url_tempi_base,index_col='Versione')


# #### Grafico sunburst del take rate

# In[5]:


fig = px.sunburst(df_take_rate,
                 path=['Versione','Optional'],
                 values = '%_Optional',
                 color='%_Optional',
                 color_continuous_scale="reds",
                 title = 'Take rate',
                 width=850, height=750)
fig.update_layout(paper_bgcolor="rgb(0,0,0,0)",
                 font_color="white")


# In[19]:


st.title('MTSV4 MTO Factory Feasibility')
st.write('Database delta tempi bundle')
st.write(df_tempi_opt)


# In[7]:


st.plotly_chart(fig, theme=None, use_container_width=True)


# #### Test grafico con slider

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


st.pyplot(fig,use_container_width=True)


# ---

# #### Orario di lavoro

# In[13]:


# Giorni_Lavorativi = 21.9
# Ore_giorno = 8


# In[14]:


Giorni_Lavorativi = st.number_input('Giorni lavorativi nel mese')
st.write('I giorni lavorativi sono: ', Giorni_Lavorativi)


# In[15]:


Ore_giorno = st.number_input('Ore / giorno')
st.write('Le ore / giorno sono: ', Ore_giorno)


# In[16]:


Tempo_disponibile = Giorni_Lavorativi*Ore_giorno*60 # uguale a Takt Time (1 moto)


# In[17]:


st.write('Il tempo disponibile nel mese è di [min] ',Tempo_disponibile)


# In[ ]:




