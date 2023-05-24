#!/usr/bin/env python
# coding: utf-8

# In[190]:


import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import requests
import plotly.express as px
from IPython.display import display
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# righe sotto inserite altrimenti non sempre mostra plotly charts
#import plotly.io as pio
#pio.renderers.default='notebook'


# #### Importazione tabelle da GitHub

# In[191]:


@st.cache_data
def load_data(url, index=''):
    if index:
        data = pd.read_excel(url, index_col=index)
    else:
        data = pd.read_excel(url)

    return data


# In[192]:


url_tempi_opt = 'https://github.com/MarcelloGalimberti/DMH/blob/main/db_tempi_opt_reale.xlsx?raw=true'


# In[193]:


df_tempi_opt = load_data(url_tempi_opt,index = 'Bundle')


# In[194]:


url_take_rate = 'https://github.com/MarcelloGalimberti/DMH/blob/main/db_take_rate.xlsx?raw=true'


# In[195]:


df_take_rate = load_data(url_take_rate)


# In[196]:


url_tempi_base = 'https://github.com/MarcelloGalimberti/DMH/blob/main/db_tempi_base.xlsx?raw=true'


# In[197]:


tempi_base = load_data(url_tempi_base,index='Versione')


# In[198]:


url_immagine = 'https://github.com/MarcelloGalimberti/DMH/blob/main/Ducati-Multistrada-V4-2021-008.jpg?raw=true'


# ---

# #### Grafico donut del take rate

# In[199]:


df_take_rate_donut = df_take_rate.copy()


# In[200]:


df_pct_Versione = df_take_rate_donut.groupby(['Versione']).mean().loc[:,['%_Versione']]*100


# In[201]:


df_pct_Versione.reset_index(inplace=True)


# In[202]:


fig5 = px.pie(df_pct_Versione, values='%_Versione', names='Versione', title='Take rate versioni',
             color_discrete_sequence=px.colors.sequential.RdBu,
             labels='Versione',
             hole=.7)
fig5.update_layout(paper_bgcolor="rgb(0,0,0,0)",
                 font_color="white")
fig5.update_traces(hovertemplate='Label: %{label}<br>Value: %{value:0.f}')
fig5.show()


# In[34]:


pivot = pd.pivot_table(df_take_rate,
                      index='Versione',
                      columns='Optional',
                      values='%_Optional',
                      aggfunc='sum',
                      fill_value=0,
                      margins=False)


# In[35]:


pivot = pivot*100


# In[36]:


pivot = pivot.astype(int)


# In[38]:


pivot # mostrare in streamlit assieme a donut chart


# #### Layout app

# In[39]:


st.image(url_immagine)


# In[40]:


st.title('MTSV4 MTO Factory Feasibility')
st.write('Database delta tempi bundle | fonte OIN')
st.write(df_tempi_opt)


# In[186]:


# Donut


# In[185]:


st.plotly_chart(fig5, theme=None, use_container_width=True)


# In[189]:


# Tabella % Take rate Bundle


# In[188]:


st.write(pivot)


# ---

# #### Orario di lavoro e veicoli/mese (input utente)

# In[42]:


Giorni_Lavorativi = st.number_input('Giorni lavorativi nel mese')
st.write('I giorni lavorativi sono: ', Giorni_Lavorativi)


# In[43]:


Ore_giorno = st.number_input('Ore / giorno')
st.write('Le ore / giorno sono: ', Ore_giorno)


# In[44]:


# Tenere per sviluppo locale
#Giorni_Lavorativi = 21.9
#Ore_giorno = 8


# In[45]:


Tempo_disponibile = Giorni_Lavorativi*Ore_giorno*60 # uguale a Takt Time (1 moto)


# In[46]:


st.write('Il tempo disponibile nel mese è di [min] ',Tempo_disponibile)


# In[47]:


Veicoli_mese = st.number_input('Veicoli / mese')
st.write('I veicoli / mese sono: ', Veicoli_mese)


# In[48]:


# Tenere per sviluppo locale
#Veicoli_mese=1482


# ---

# ## Processo elaborazione dati

# #### Database take rate con volumi mese

# In[49]:


#@st.cache_data RIPROVARE A DECORARE
def take_rate (Veicoli_mese):
    df_take_rate['Qty versione']=(df_take_rate['%_Versione']*Veicoli_mese).round().astype(int)
    df_take_rate['Qty optional']=(df_take_rate['%_Optional']*df_take_rate['Qty versione']).round().astype(int)
    return df_take_rate


# In[50]:


take_rate(Veicoli_mese) # dato da input in Streamlit altrimenti 1482


# #### Database take rate per ogni versione

# In[54]:


df_PP = df_take_rate[(df_take_rate['Versione'] == 'Pikes Peak')]
df_Rally = df_take_rate[(df_take_rate['Versione'] == 'Rally')]
df_RS = df_take_rate[(df_take_rate['Versione'] == 'RS')]
df_S = df_take_rate[(df_take_rate['Versione'] == 'S')]
df_S_Grand_Tour = df_take_rate[(df_take_rate['Versione'] == 'S Grand Tour')]
df_Standard = df_take_rate[(df_take_rate['Versione'] == 'Standard')]


# #### Funzione: df vuoto in base alla versione (es: arg = df_PP)

# In[55]:


#@st.cache_data RIPROVARE A DECORARE
def crea_df_veicoli (df): # inserire ad esempio df_PP
    lista_optional = list(df['Optional'].unique())
    df_veicoli_creato = pd.DataFrame(columns =  lista_optional, index = range(df['Qty versione'].max()))
    return df_veicoli_creato


# In[56]:


df_veicoli_PP = crea_df_veicoli(df_PP)
df_veicoli_Rally = crea_df_veicoli(df_Rally)
df_veicoli_RS = crea_df_veicoli(df_RS)
df_veicoli_S = crea_df_veicoli(df_S)
df_veicoli_S_Grand_Tour = crea_df_veicoli(df_S_Grand_Tour)
df_veicoli_Standard = crea_df_veicoli(df_Standard)


# #### Funzione: compila il database vuoto con il numero di opt per veicolo
# es: arg (df_PP, df_veicoli_PP)

# In[57]:


#@st.cache_data RIPROVARE A DECORARE
def compila_veicoli (df,df_veicoli):
    for i in range (len(df)):
        for j in range (len(df_veicoli)):
            if j < df.iloc[i]['Qty optional']:
                df_veicoli.iat[j,i] = 1
            else:
                df_veicoli.iat[j,i] = 0
    df_veicoli.index.name='id_veicolo'
    return df_veicoli


# In[58]:


df_veicoli_PP = compila_veicoli (df_PP, df_veicoli_PP)
df_veicoli_Rally = compila_veicoli (df_Rally,df_veicoli_Rally)
df_veicoli_RS = compila_veicoli (df_RS,df_veicoli_RS)
df_veicoli_S = compila_veicoli (df_S,df_veicoli_S)
df_veicoli_S_Grand_Tour = compila_veicoli (df_S_Grand_Tour,df_veicoli_S_Grand_Tour)
df_veicoli_Standard = compila_veicoli (df_Standard,df_veicoli_Standard)


# #### Funzione: shuffle per ogni colonna del df_veicoli con opt

# In[59]:


#@st.cache_data RIPROVARE A DECORARE
def shuffle_veicoli (df_veicoli):
    for i in df_veicoli.columns:
        df_veicoli[i]=df_veicoli[i].sample(frac=1,random_state=1).values
    return df_veicoli


# In[60]:


df_veicoli_PP = shuffle_veicoli(df_veicoli_PP)
df_veicoli_Rally = shuffle_veicoli(df_veicoli_Rally)
df_veicoli_RS = shuffle_veicoli(df_veicoli_RS)
df_veicoli_S = shuffle_veicoli(df_veicoli_S)
df_veicoli_S_Grand_Tour = shuffle_veicoli(df_veicoli_S_Grand_Tour)
df_veicoli_Standard = shuffle_veicoli(df_veicoli_Standard)


# #### Funzione: carica i tempi dei veicoli con opt ** LUNGO RUNTIME **
# es: arg(df_veicoli_PP, df_delta_t)
# 

# In[61]:


@st.cache_data
def carica_delta_t(df_veicoli):
    lista_indici=[]
    df_delta_t = pd.DataFrame(columns=['id_veicolo','stgr', 'linea', 'collaudo', 'radar', 'vestizione'])
    df_delta_t.set_index('id_veicolo', inplace=True)
    
    for i in range (len(df_veicoli)):
        for j in range (len(df_veicoli.columns)):
            if df_veicoli.iloc[i][j] == 1:
                df_delta_t.loc[len(df_delta_t)] = df_tempi_opt.loc[df_veicoli.columns[j]]
                lista_indici.append(df_delta_t.index[i])
            else:
                df_delta_t.loc[len(df_delta_t)] = 0
                lista_indici.append(df_delta_t.index[i])
    for i in range(len(lista_indici)):
        df_delta_t.rename(index={i:lista_indici[i]}, inplace=True)
    df_delta_t_per_veicolo = df_delta_t.groupby('id_veicolo').sum()
    return df_delta_t_per_veicolo


# #### Creazione e modifica dataframe **creare funzioni**

# In[62]:


df_delta_tempi_PP = carica_delta_t(df_veicoli_PP)
df_delta_tempi_Rally = carica_delta_t(df_veicoli_Rally)
df_delta_tempi_RS = carica_delta_t(df_veicoli_RS)
df_delta_tempi_S = carica_delta_t(df_veicoli_S)
df_delta_tempi_S_Grand_Tour = carica_delta_t(df_veicoli_S_Grand_Tour)
df_delta_tempi_Standard = carica_delta_t(df_veicoli_Standard)


# In[63]:


df_delta_tempi_PP['Versione']='Pikes Peak'
df_delta_tempi_Rally['Versione'] = 'Rally'
df_delta_tempi_RS['Versione'] = 'RS'
df_delta_tempi_S['Versione'] = 'S'
df_delta_tempi_S_Grand_Tour['Versione'] = 'S Grand Tour'
df_delta_tempi_Standard['Versione'] = 'Standard'


# In[64]:


df_delta_tempi_PP.set_index('Versione',inplace=True)
df_delta_tempi_Rally.set_index('Versione',inplace=True)
df_delta_tempi_RS.set_index('Versione',inplace=True)
df_delta_tempi_S.set_index('Versione',inplace=True)
df_delta_tempi_S_Grand_Tour.set_index('Versione',inplace=True)
df_delta_tempi_Standard.set_index('Versione',inplace=True)


# In[65]:


df_delta_tempi_PP = df_delta_tempi_PP.add(tempi_base)
df_delta_tempi_Rally = df_delta_tempi_Rally.add(tempi_base)
df_delta_tempi_RS = df_delta_tempi_RS.add(tempi_base)
df_delta_tempi_S = df_delta_tempi_S.add(tempi_base)
df_delta_tempi_S_Grand_Tour = df_delta_tempi_S_Grand_Tour.add(tempi_base)
df_delta_tempi_Standard = df_delta_tempi_Standard.add(tempi_base)


# In[66]:


df_delta_tempi_PP.dropna(inplace = True)
df_delta_tempi_Rally.dropna(inplace = True)
df_delta_tempi_RS.dropna(inplace = True)
df_delta_tempi_S.dropna(inplace = True)
df_delta_tempi_S_Grand_Tour.dropna(inplace = True)
df_delta_tempi_Standard.dropna(inplace = True)


# In[67]:


# QUI RESETTARE INDICE


# In[68]:


df_delta_tempi_PP.reset_index(inplace=True)
df_delta_tempi_Rally.reset_index(inplace=True)
df_delta_tempi_RS.reset_index(inplace=True)
df_delta_tempi_S.reset_index(inplace=True)
df_delta_tempi_S_Grand_Tour.reset_index(inplace=True)
df_delta_tempi_Standard.reset_index(inplace=True)


# ---

# In[69]:


# vertical_concat serve?


# In[70]:


# vertical_concat = pd.concat([df_delta_tempi_PP,
#                             df_delta_tempi_Rally,
#                            df_delta_tempi_RS,
#                            df_delta_tempi_S,
#                            df_delta_tempi_S_Grand_Tour,
#                            df_delta_tempi_Standard], axis=0)


# #### Unione database veicoli **creare funzione e decorare**

# In[71]:


df_delta_t_opt_PP = pd.concat([df_delta_tempi_PP,df_veicoli_PP], axis=1)
df_delta_t_opt_Rally = pd.concat([df_delta_tempi_Rally,df_veicoli_Rally], axis=1)
df_delta_t_opt_RS = pd.concat([df_delta_tempi_RS,df_veicoli_RS], axis=1)
df_delta_t_opt_S = pd.concat([df_delta_tempi_S,df_veicoli_S], axis=1)
df_delta_t_opt_S_Grand_Tour = pd.concat([df_delta_tempi_S_Grand_Tour,df_veicoli_S_Grand_Tour], axis=1)                              
df_delta_t_opt_Standard = pd.concat([df_delta_tempi_Standard,df_veicoli_Standard], axis=1)


# In[72]:


unione_veicoli = pd.concat([df_delta_t_opt_PP,
                            df_delta_t_opt_Rally,
                            df_delta_t_opt_RS,
                            df_delta_t_opt_S,
                            df_delta_t_opt_S_Grand_Tour,
                            df_delta_t_opt_Standard,
                           ], ignore_index= True,axis = 0).fillna(0).set_index('Versione')


# In[73]:


unione_veicoli_caratteristiche = unione_veicoli.copy()


# In[74]:


unione_veicoli_caratteristiche


# In[75]:


# QUI VISULIZZARE ED ESPORTARE ORDINI SIMULATI


# In[76]:


unione_veicoli.drop(unione_veicoli.columns[5:21], inplace=True, axis=1)


# In[77]:


# unione_veicoli #altrinenti viene visualizzato in streamlit


# #### Fabbisogno risorse

# In[78]:


fabbisogno_risorse = unione_veicoli.copy()


# In[79]:


fabbisogno_risorse = fabbisogno_risorse/Tempo_disponibile


# In[80]:


df_fabbisogno = pd.DataFrame(fabbisogno_risorse.sum()).T


# In[81]:


df_fabbisogno['Origine']='Fabbisogno'


# In[82]:


df_fabbisogno.set_index(['Origine'], inplace=True)


# In[83]:


st.write('Fabbisogno risorse')
st.write(df_fabbisogno)


# In[84]:


df_fabbisogno


# #### Confronto tra fabbisogno e risorse disponibili (input slider)

# In[85]:


st.write('Inserire capacità installata')


# In[86]:


stgr_installati = st.slider('stgr installati',1,50,(25))
linea_installati = st.slider('linea installati',1,50,(43))
collaudo_installati = st.slider('collaudo installati',1,20,(12))
radar_installati = st.slider('radar installati',1,2,(1))
vestizione_installati = st.slider('vestizione installati',1,20,(20))


# In[87]:


# Tenere per sviluppo in locale
#stgr_installati = 25
#linea_installati = 43
#collaudo_installati = 12
#radar_installati = 1
#vestizione_installati = 20


# In[88]:


colonne_v_p = ['Origine', 'stgr','linea','collaudo','radar','vestizione']


# In[89]:


numero_risorse = ['Capacità installata',
                  stgr_installati,
                  linea_installati,
                  collaudo_installati,
                  radar_installati,
                  vestizione_installati]


# In[90]:


df_test = pd.DataFrame(columns=colonne_v_p)


# In[91]:


df_test.loc[len(df_test)] = numero_risorse


# In[92]:


df_test.set_index('Origine',inplace=True)


# In[93]:


vincoli_produzione = df_test.copy()


# In[94]:


df_confronto = pd.concat([vincoli_produzione,df_fabbisogno],axis=0)


# In[95]:


st.write('Fabbisogno vs Capacità')
st.write(df_confronto)


# In[96]:


df_confronto


# #### Calcolo saturazione

# In[97]:


saturazione = []
for i in range(len(df_confronto.columns)):
    sat = (df_confronto.iloc[1][i]/df_confronto.iloc[0][i])
    saturazione.append(sat)


# In[98]:


lista_colonne_saturazione = list(df_confronto.columns)


# In[99]:


df_saturazione = pd.DataFrame(saturazione)


# In[100]:


df_saturazione=df_saturazione.T


# In[101]:


df_saturazione.columns = lista_colonne_saturazione


# #### Bar chart saturazione

# In[102]:


df_chart = df_saturazione * 100

# Set the seaborn style and palette
sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("Reds_r"))

# Create the bar chart
fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figsize as needed
ax = sns.barplot(data=df_chart)

for i in ax.containers:
    ax.bar_label(i)

# Set the title
ax.set_title("Saturazione [%]")

# Display the chart in Streamlit
st.pyplot(fig,use_container_width=True)


# ---

# #### Creazione id veicolo

# In[103]:


unione_veicoli_caratteristiche.reset_index(inplace=True)


# In[104]:


unione_veicoli_caratteristiche['C'] = unione_veicoli_caratteristiche.groupby('Versione').cumcount()+1


# In[105]:


unione_veicoli_caratteristiche['C']=unione_veicoli_caratteristiche['C'].astype(str)


# In[106]:


colonne_id = ['Versione','C']
unione_veicoli_caratteristiche['id']=unione_veicoli_caratteristiche[colonne_id].apply(lambda x: '|'.join(x), axis=1)


# In[107]:


colonne_tempi_ciclo = ['stgr','linea','collaudo','radar','vestizione']


# In[108]:


unione_veicoli_caratteristiche['TCT']=unione_veicoli_caratteristiche[colonne_tempi_ciclo].sum(axis=1)


# In[109]:


unione_veicoli_caratteristiche['CV']=unione_veicoli_caratteristiche[colonne_tempi_ciclo].std(axis=1)/unione_veicoli_caratteristiche[colonne_tempi_ciclo].mean(axis=1)


# #### Clustering su TCT e CV

# In[110]:


X = unione_veicoli_caratteristiche.loc[:,['TCT','CV']] # tempi ciclo fasi principali


# In[111]:


X


# In[112]:


scaler = StandardScaler()


# In[113]:


# fit and transform the data
scaled_data = scaler.fit_transform(X) 


# In[114]:


# definisco istanza di KMeans
km = KMeans(n_clusters = 2,
           init ='k-means++',
           n_init=10,
           max_iter=300,
           tol=1e-04,
           random_state=0)


# In[115]:


clusters=km.fit_predict(scaled_data)


# In[116]:


df_describe = pd.DataFrame(clusters)
df_describe.value_counts()


# In[117]:


unione_veicoli_caratteristiche=pd.concat([unione_veicoli_caratteristiche,df_describe],axis=1)


# In[118]:


unione_veicoli_caratteristiche.rename(columns = {0:'Cluster'}, inplace = True)


# In[119]:


unione_veicoli_caratteristiche


# #### Assegnazione criticità in base all'appartenenza cluster

# In[120]:


unione_veicoli_caratteristiche['critico']=unione_veicoli_caratteristiche['Cluster']==0


# In[121]:


unione_veicoli_caratteristiche.Cluster.value_counts()


# #### Processo di selezione liste heijunka

# In[122]:


lista_heihunka = []
lista_nc = []
lista_c = []


# In[123]:


for i in range (len(unione_veicoli_caratteristiche)):
    if unione_veicoli_caratteristiche.iloc[i]['critico']==True:
        lista_c.append(unione_veicoli_caratteristiche.loc[i]['id'])
    else:
        lista_nc.append(unione_veicoli_caratteristiche.loc[i]['id'])


# In[124]:


lista_heihunka=[lista_c,lista_nc]


# #### Algoritmo heijunka

# In[125]:


count = sum([ len(listElem) for listElem in lista_heihunka])


# In[126]:


sequenza = list(range(count))
residuo = list(range(count))
for i in range(len(lista_heihunka)):
    if i == 0:
        indice_versione = []
        indice_versione = np.linspace(0,len(residuo)-1,len(lista_heihunka[i]),dtype=int)
        counter = 0
    
        for k in indice_versione:
            sequenza[k]=lista_heihunka[i][counter]
            counter = counter +1
    else:
        residuo = np.delete(residuo,indice_versione)
        counter = 0
        
        for k in residuo:
            sequenza[k]=lista_heihunka[i][counter]
            counter = counter +1


# #### Crezione sequenza veicoli heijunka

# In[127]:


d = {'Sequenza': sequenza}
df_heijunka = pd.DataFrame(d)


# In[128]:


df_heijunka_ind = df_heijunka.copy()


# In[129]:


df_heijunka_ind['id']=df_heijunka_ind['Sequenza']


# In[130]:


df_heijunka_ind.set_index('id',inplace=True)


# In[131]:


df_uvc = unione_veicoli_caratteristiche.copy()


# In[132]:


df_uvc.set_index('id',inplace=True)


# In[133]:


df_sequenza = pd.concat([df_heijunka_ind,df_uvc],axis = 1)


# In[134]:


df_sequenza


# #### Visualizzare sequenza

# In[135]:


df_sequenza.to_excel('/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Ducati 2023/ASC/Simulazione/df_sequenza_cluster_2.xlsx')


# In[136]:


st.dataframe(df_sequenza)


# In[137]:


df_sequenza


# #### Download file

# In[ ]:


#@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df_sequenza)

st.download_button(
    label="Download sequenza in CSV",
    data=csv,
    file_name='sequenza.csv',
    mime='text/csv',
)


# #### Grafico 1 giorno

# In[ ]:


df_grafico_sequenza = df_sequenza.iloc[0:31]


# In[ ]:


st.bar_chart(data=df_grafico_sequenza, y='TCT',use_container_width=True)


# In[ ]:




