#!/usr/bin/env python
# coding: utf-8

# In[108]:


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
import plotly.io as pio
pio.renderers.default='notebook'
from sklearn.mixture import GaussianMixture
from sklearn. decomposition import PCA


# #### Importazione tabelle da GitHub

# In[2]:


@st.cache_data
def load_data(url, index=''):
    if index:
        data = pd.read_excel(url, index_col=index)
    else:
        data = pd.read_excel(url)

    return data


# In[3]:


url_tempi_opt = 'https://github.com/MarcelloGalimberti/DMH/blob/main/db_tempi_opt_reale.xlsx?raw=true'


# In[4]:


df_tempi_opt = load_data(url_tempi_opt,index = 'Bundle')


# In[5]:


url_take_rate = 'https://github.com/MarcelloGalimberti/DMH/blob/main/db_take_rate.xlsx?raw=true'


# In[6]:


df_take_rate = load_data(url_take_rate)


# In[7]:


url_tempi_base = 'https://github.com/MarcelloGalimberti/DMH/blob/main/db_tempi_base.xlsx?raw=true'


# In[8]:


tempi_base = load_data(url_tempi_base,index='Versione')


# In[9]:


url_immagine = 'https://github.com/MarcelloGalimberti/DMH/blob/main/Ducati-Multistrada-V4-2021-008.jpg?raw=true'


# In[10]:


# tempi_base


# In[11]:


# df_tempi_opt


# ---

# #### Grafico sunburst del take rate

# In[ ]:


# inner_mean = np.mean(df_take_rate['%_Versione'])
fig = px.sunburst(df_take_rate,
                 path=['Versione','Optional'],
                 values = '%_Optional',
                 color='%_Versione',
                 color_continuous_scale="reds",
                 title = 'Take rate',
                 width=850, height=750
                 
                 
                 )
fig.update_layout(paper_bgcolor="rgb(0,0,0,0)",
                 font_color="white")
#fig.update_traces(textinfo="label+percent entry")
#fig.show()


# In[14]:


pivot = pd.pivot_table(df_take_rate,
                      index='Versione',
                      columns='Optional',
                      values='%_Optional',
                      aggfunc='sum',
                      fill_value=0,
                      margins=False)


# In[15]:


pivot = pivot*100


# In[16]:


pivot = pivot.astype(int)


# In[17]:


#pivot # mostrare in streamlit assieme a donut chart


# #### Layout app

# In[18]:


st.image(url_immagine)


# In[19]:


st.title('MTSV4 MTO Factory Feasibility')
st.write('Database delta tempi bundle | fonte OIN')
st.write(df_tempi_opt)


# In[20]:


# Sunburst


# In[21]:


st.plotly_chart(fig, theme=None, use_container_width=True)


# In[22]:


# st.dataframe(pivot) # sistemare output


# ---

# #### Orario di lavoro e veicoli/mese (input utente)

# In[24]:


Giorni_Lavorativi = st.number_input('Giorni lavorativi nel mese')
st.write('I giorni lavorativi sono: ', Giorni_Lavorativi)


# In[25]:


Ore_giorno = st.number_input('Ore / giorno')
st.write('Le ore / giorno sono: ', Ore_giorno)


# In[154]:


# Tenere per sviluppo locale
#Giorni_Lavorativi = 22 # ex 21.9
#Ore_giorno = 8


# In[27]:


Tempo_disponibile = Giorni_Lavorativi*Ore_giorno*60 # uguale a Takt Time (1 moto)


# In[28]:


st.write('Il tempo disponibile nel mese è di [min] ',Tempo_disponibile)


# In[29]:


Veicoli_mese = st.number_input('Veicoli / mese')
st.write('I veicoli / mese sono: ', Veicoli_mese)


# In[155]:


# Tenere per sviluppo locale
#Veicoli_mese=1482


# ---

# ## Processo elaborazione dati

# #### Database take rate con volumi mese

# In[31]:


#@st.cache_data RIPROVARE A DECORARE | ARROTONDAMENTO
def take_rate (Veicoli_mese):
    df_take_rate['Qty versione']=(df_take_rate['%_Versione']*Veicoli_mese).round().astype(int)
    df_take_rate['Qty optional']=(df_take_rate['%_Optional']*df_take_rate['Qty versione']).round().astype(int)
    return df_take_rate


# In[32]:


take_rate(Veicoli_mese) # dato da input in Streamlit altrimenti 1482


# #### Database take rate per ogni versione

# In[33]:


df_PP = df_take_rate[(df_take_rate['Versione'] == 'Pikes Peak')]
df_Rally = df_take_rate[(df_take_rate['Versione'] == 'Rally')]
df_RS = df_take_rate[(df_take_rate['Versione'] == 'RS')]
df_S = df_take_rate[(df_take_rate['Versione'] == 'S')]
df_S_Grand_Tour = df_take_rate[(df_take_rate['Versione'] == 'S Grand Tour')]
df_Standard = df_take_rate[(df_take_rate['Versione'] == 'Standard')]


# #### Funzione: df vuoto in base alla versione (es: arg = df_PP)

# In[34]:


#@st.cache_data RIPROVARE A DECORARE
def crea_df_veicoli (df): # inserire ad esempio df_PP
    lista_optional = list(df['Optional'].unique())
    df_veicoli_creato = pd.DataFrame(columns =  lista_optional, index = range(df['Qty versione'].max()))
    return df_veicoli_creato


# In[35]:


df_veicoli_PP = crea_df_veicoli(df_PP)
df_veicoli_Rally = crea_df_veicoli(df_Rally)
df_veicoli_RS = crea_df_veicoli(df_RS)
df_veicoli_S = crea_df_veicoli(df_S)
df_veicoli_S_Grand_Tour = crea_df_veicoli(df_S_Grand_Tour)
df_veicoli_Standard = crea_df_veicoli(df_Standard)


# #### Funzione: compila il database vuoto con il numero di opt per veicolo
# es: arg (df_PP, df_veicoli_PP)

# In[36]:


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


# In[37]:


df_veicoli_PP = compila_veicoli (df_PP, df_veicoli_PP)
df_veicoli_Rally = compila_veicoli (df_Rally,df_veicoli_Rally)
df_veicoli_RS = compila_veicoli (df_RS,df_veicoli_RS)
df_veicoli_S = compila_veicoli (df_S,df_veicoli_S)
df_veicoli_S_Grand_Tour = compila_veicoli (df_S_Grand_Tour,df_veicoli_S_Grand_Tour)
df_veicoli_Standard = compila_veicoli (df_Standard,df_veicoli_Standard)


# #### Funzione: shuffle per ogni colonna del df_veicoli con opt

# In[38]:


#@st.cache_data RIPROVARE A DECORARE
def shuffle_veicoli (df_veicoli):
    for i in df_veicoli.columns:
        df_veicoli[i]=df_veicoli[i].sample(frac=1,random_state=1).values
    return df_veicoli


# In[39]:


df_veicoli_PP = shuffle_veicoli(df_veicoli_PP)
df_veicoli_Rally = shuffle_veicoli(df_veicoli_Rally)
df_veicoli_RS = shuffle_veicoli(df_veicoli_RS)
df_veicoli_S = shuffle_veicoli(df_veicoli_S)
df_veicoli_S_Grand_Tour = shuffle_veicoli(df_veicoli_S_Grand_Tour)
df_veicoli_Standard = shuffle_veicoli(df_veicoli_Standard)


# #### Funzione: carica i tempi dei veicoli con opt ** LUNGO RUNTIME **
# es: arg(df_veicoli_PP, df_delta_t)
# 

# In[40]:


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

# In[41]:


df_delta_tempi_PP = carica_delta_t(df_veicoli_PP)
df_delta_tempi_Rally = carica_delta_t(df_veicoli_Rally)
df_delta_tempi_RS = carica_delta_t(df_veicoli_RS)
df_delta_tempi_S = carica_delta_t(df_veicoli_S)
df_delta_tempi_S_Grand_Tour = carica_delta_t(df_veicoli_S_Grand_Tour)
df_delta_tempi_Standard = carica_delta_t(df_veicoli_Standard)


# In[42]:


df_delta_tempi_PP['Versione']='Pikes Peak'
df_delta_tempi_Rally['Versione'] = 'Rally'
df_delta_tempi_RS['Versione'] = 'RS'
df_delta_tempi_S['Versione'] = 'S'
df_delta_tempi_S_Grand_Tour['Versione'] = 'S Grand Tour'
df_delta_tempi_Standard['Versione'] = 'Standard'


# In[43]:


df_delta_tempi_PP.set_index('Versione',inplace=True)
df_delta_tempi_Rally.set_index('Versione',inplace=True)
df_delta_tempi_RS.set_index('Versione',inplace=True)
df_delta_tempi_S.set_index('Versione',inplace=True)
df_delta_tempi_S_Grand_Tour.set_index('Versione',inplace=True)
df_delta_tempi_Standard.set_index('Versione',inplace=True)


# In[44]:


df_delta_tempi_PP = df_delta_tempi_PP.add(tempi_base)
df_delta_tempi_Rally = df_delta_tempi_Rally.add(tempi_base)
df_delta_tempi_RS = df_delta_tempi_RS.add(tempi_base)
df_delta_tempi_S = df_delta_tempi_S.add(tempi_base)
df_delta_tempi_S_Grand_Tour = df_delta_tempi_S_Grand_Tour.add(tempi_base)
df_delta_tempi_Standard = df_delta_tempi_Standard.add(tempi_base)


# In[45]:


df_delta_tempi_PP.dropna(inplace = True)
df_delta_tempi_Rally.dropna(inplace = True)
df_delta_tempi_RS.dropna(inplace = True)
df_delta_tempi_S.dropna(inplace = True)
df_delta_tempi_S_Grand_Tour.dropna(inplace = True)
df_delta_tempi_Standard.dropna(inplace = True)


# In[46]:


# QUI RESETTARE INDICE


# In[47]:


df_delta_tempi_PP.reset_index(inplace=True)
df_delta_tempi_Rally.reset_index(inplace=True)
df_delta_tempi_RS.reset_index(inplace=True)
df_delta_tempi_S.reset_index(inplace=True)
df_delta_tempi_S_Grand_Tour.reset_index(inplace=True)
df_delta_tempi_Standard.reset_index(inplace=True)


# ---

# #### Unione database veicoli **creare funzione e decorare**

# In[48]:


df_delta_t_opt_PP = pd.concat([df_delta_tempi_PP,df_veicoli_PP], axis=1)
df_delta_t_opt_Rally = pd.concat([df_delta_tempi_Rally,df_veicoli_Rally], axis=1)
df_delta_t_opt_RS = pd.concat([df_delta_tempi_RS,df_veicoli_RS], axis=1)
df_delta_t_opt_S = pd.concat([df_delta_tempi_S,df_veicoli_S], axis=1)
df_delta_t_opt_S_Grand_Tour = pd.concat([df_delta_tempi_S_Grand_Tour,df_veicoli_S_Grand_Tour], axis=1)                              
df_delta_t_opt_Standard = pd.concat([df_delta_tempi_Standard,df_veicoli_Standard], axis=1)


# In[49]:


unione_veicoli = pd.concat([df_delta_t_opt_PP,
                            df_delta_t_opt_Rally,
                            df_delta_t_opt_RS,
                            df_delta_t_opt_S,
                            df_delta_t_opt_S_Grand_Tour,
                            df_delta_t_opt_Standard,
                           ], ignore_index= True,axis = 0).fillna(0).set_index('Versione')


# In[50]:


unione_veicoli_caratteristiche = unione_veicoli.copy()


# In[51]:


st.write('Ordini simulati nel mese')
st.write(unione_veicoli_caratteristiche)


# In[52]:


#unione_veicoli_caratteristiche


# In[53]:


#unione_veicoli_caratteristiche.to_excel('/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Ducati 2023/ASC/Simulazione/uvc_per_cluster.xlsx')


# In[54]:


# QUI VISULIZZARE ED ESPORTARE ORDINI SIMULATI


# In[55]:


unione_veicoli.drop(unione_veicoli.columns[5:21], inplace=True, axis=1)


# In[56]:


# unione_veicoli #altrinenti viene visualizzato in streamlit


# In[57]:


# serve per input file calcolo cluster


# In[58]:


df_uvc_per_cluster = unione_veicoli.reset_index()


# In[59]:


# df_uvc_per_cluster.to_excel('/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Ducati 2023/ASC/Simulazione/uvc_per_cluster.xlsx')


# #### Fabbisogno risorse

# In[60]:


fabbisogno_risorse = unione_veicoli.copy()


# In[61]:


fabbisogno_risorse = fabbisogno_risorse/Tempo_disponibile


# In[62]:


df_fabbisogno = pd.DataFrame(fabbisogno_risorse.sum()).T


# In[63]:


df_fabbisogno['Origine']='Fabbisogno'


# In[64]:


df_fabbisogno.set_index(['Origine'], inplace=True)


# In[65]:


st.write('Fabbisogno risorse')
st.write(df_fabbisogno)


# In[153]:


#df_fabbisogno


# #### Confronto tra fabbisogno e risorse disponibili (input slider)

# In[67]:


st.write('Inserire capacità installata')


# In[70]:


stgr_installati = st.slider('stgr installati',1,50,(30))
linea_installati = st.slider('linea installati',1,50,(43))
collaudo_installati = st.slider('collaudo installati',1,20,(12))
radar_installati = st.slider('radar installati',1,2,(2))
vestizione_installati = st.slider('vestizione installati',1,20,(20))


# In[71]:


# Tenere per sviluppo in locale
stgr_installati = 30
linea_installati = 43
collaudo_installati = 12
radar_installati = 2
vestizione_installati = 20


# In[72]:


colonne_v_p = ['Origine', 'stgr','linea','collaudo','radar','vestizione']


# In[73]:


numero_risorse = ['Capacità installata',
                  stgr_installati,
                  linea_installati,
                  collaudo_installati,
                  radar_installati,
                  vestizione_installati]


# In[74]:


df_test = pd.DataFrame(columns=colonne_v_p)


# In[75]:


df_test.loc[len(df_test)] = numero_risorse


# In[76]:


df_test.set_index('Origine',inplace=True)


# In[77]:


vincoli_produzione = df_test.copy()


# In[78]:


df_confronto = pd.concat([vincoli_produzione,df_fabbisogno],axis=0)


# In[79]:


st.write('Fabbisogno vs Capacità')
st.write(df_confronto)


# In[80]:


#df_confronto


# #### Calcolo saturazione

# In[81]:


saturazione = []
for i in range(len(df_confronto.columns)):
    sat = (df_confronto.iloc[1][i]/df_confronto.iloc[0][i])
    saturazione.append(sat)


# In[82]:


lista_colonne_saturazione = list(df_confronto.columns)


# In[83]:


df_saturazione = pd.DataFrame(saturazione)


# In[84]:


df_saturazione=df_saturazione.T


# In[85]:


df_saturazione.columns = lista_colonne_saturazione


# In[86]:


#df_saturazione


# #### Bar chart saturazione

# In[87]:


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

# In[90]:


unione_veicoli_caratteristiche.reset_index(inplace=True)


# In[91]:


unione_veicoli_caratteristiche['C'] = unione_veicoli_caratteristiche.groupby('Versione').cumcount()+1


# In[92]:


unione_veicoli_caratteristiche['C']=unione_veicoli_caratteristiche['C'].astype(str)


# In[93]:


colonne_id = ['Versione','C']
unione_veicoli_caratteristiche['id']=unione_veicoli_caratteristiche[colonne_id].apply(lambda x: '|'.join(x), axis=1)


# In[94]:


colonne_tempi_ciclo = ['stgr','linea','collaudo','radar','vestizione']


# In[95]:


unione_veicoli_caratteristiche['TCT']=unione_veicoli_caratteristiche[colonne_tempi_ciclo].sum(axis=1)


# In[96]:


unione_veicoli_caratteristiche['CV']=unione_veicoli_caratteristiche[colonne_tempi_ciclo].std(axis=1)/unione_veicoli_caratteristiche[colonne_tempi_ciclo].mean(axis=1)


# #### PCA

# In[97]:


# unione_veicoli_caratteristiche


# In[98]:


X_valori = unione_veicoli_caratteristiche[colonne_tempi_ciclo]


# In[99]:


#X_valori


# In[100]:


scaler = StandardScaler()


# In[101]:


X_valori=scaler.fit_transform(X_valori)


# In[102]:


#X_valori


# In[103]:


pca_2=PCA(n_components=2,random_state=0)
X_pca_2=pca_2.fit_transform(X_valori)


# In[104]:


#X_pca_2


# In[105]:


df_describe_PCA = pd.DataFrame(X_pca_2, columns=['PCA1','PCA2'])


# In[107]:


fig = px.scatter(df_describe_PCA, x="PCA1", y="PCA2",template='plotly_dark')#,color="Cluster"
                 #,hover_data=['radar'],symbol="radar")
fig.update_traces(marker_size=10)
#fig.update_layout(barmode="group")#, scattergap=0.5)
fig.show()


# #### Clustering su PCA

# In[110]:


# caricare Yellowbrick


# In[111]:


model_km_PCA = KMeans(n_clusters = 4,
           init ='k-means++',
           n_init=10,
           max_iter=300,
           tol=1e-04,
           random_state=0)

clusters_km_PCA=model_km_PCA.fit_predict(X_pca_2)


# In[112]:


df_describe_km_PCA = pd.DataFrame(clusters_km_PCA)
df_describe_km_PCA.value_counts()


# In[113]:


df_km_PCA=pd.concat([unione_veicoli_caratteristiche,df_describe_km_PCA],axis=1)
df_km_PCA.rename(columns = {0:'Cluster'}, inplace = True)
#df_km_PCA


# In[116]:


unione_veicoli_caratteristiche=df_km_PCA


# In[117]:


#unione_veicoli_caratteristiche


# In[118]:


# assegnare colori ai cluster
fig = px.scatter(unione_veicoli_caratteristiche, x="CV", y="TCT",template='plotly_dark',
                color="Cluster",
                 hover_data=['radar'],symbol="radar")
fig.update_traces(marker_size=10)
#fig.update_layout(barmode="group")#, scattergap=0.5)
fig.show()


# #### Algoritmo heijunka per cluster

# In[119]:


df_h=unione_veicoli_caratteristiche.copy()


# In[120]:


#df_h


# In[121]:


df_conteggi = df_h['Cluster'].value_counts().rename_axis('id_cluster').reset_index(name='counts')


# In[122]:


df_conteggi=df_conteggi.astype('int')


# In[123]:


#df_conteggi casomai dare come info di analisi


# In[124]:


lunghezza_residua = len(df_h)


# In[125]:


lista_start = list(np.arange(0,len(df_h),1))


# In[126]:


for i in range(len(df_conteggi)):
    posizioni=[]
    posizioni =list(np.linspace(0,
                                lunghezza_residua-1,
                                list(df_conteggi.loc[i,['counts']])[0],
                                dtype='int'))
    lunghezza_residua = lunghezza_residua-list(df_conteggi.loc[i,['counts']])[0]
    lista_indici = list(map(lambda x: lista_start[x],posizioni))
    k=0
    numero_cluster = list(df_conteggi.loc[i,['id_cluster']])[0]
    for j in df_h[df_h['Cluster']==numero_cluster].index:
        df_h.at[j,'sequenza']=lista_indici[k]
        k=k+1
    lista_start=list(np.delete(lista_start,posizioni))


# #### Crezione sequenza veicoli heijunka

# In[128]:


df_h_sequenziato = df_h.sort_values(by=['sequenza'])


# #### Visualizzare sequenza

# In[130]:


st.write('Sequenza produttiva heijunka - dataframe')
st.dataframe(df_h_sequenziato)
#df_h_sequenziato


# In[131]:


#df_h_sequenziato.to_excel('/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Ducati 2023/ASC/Simulazione/df_h_sequenziato.xlsx')


# #### Download file

# In[133]:


#@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df_h_sequenziato)

st.download_button(
    label="Download sequenza produttiva in CSV",
    data=csv,
    file_name='sequenza.csv',
    mime='text/csv',
)


# #### Grafici

# In[169]:


fig = px.line(df_h_sequenziato, x="sequenza", y="TCT", title='Run chart',
             color='Cluster',template='plotly_dark')
fig.show()
st.plotly_chart(fig, theme=None, use_container_width=True)


# In[166]:


for j in colonne_tempi_ciclo:
    fig = px.line(df_h_sequenziato, x="sequenza", y=j, title='Run chart',
                 color='Cluster',template='plotly_dark')#,color_continuous_scale="reds")
    fig.show()
    st.plotly_chart(fig, theme=None, use_container_width=True)


# #### Calcolo FMR

# In[140]:


df_h_sequenziato_FMR = df_h_sequenziato.reset_index()


# In[142]:


#df_h_sequenziato_FMR


# In[143]:


lista_colonne_saturazione = ['stgr', 'linea', 'collaudo', 'radar', 'vestizione']
df_workload_day = pd.DataFrame(columns = lista_colonne_saturazione)


# In[144]:


lotto_moto=np.rint(len(df_h_sequenziato_FMR)/Giorni_Lavorativi).astype(int)
for start_day in range(0,
                       len(df_h_sequenziato_FMR),
                       lotto_moto):
    lista_workload = []
    for risorse in colonne_tempi_ciclo:
        lista_workload.append(df_h_sequenziato_FMR.loc[start_day:start_day+lotto_moto,risorse].sum()) 
    df_workload_day.loc[len(df_workload_day)]=lista_workload   


# In[145]:


#df_workload_day


# In[147]:


df_fmr_day=df_workload_day/480


# In[148]:


df_fmr_day=df_fmr_day.iloc[0:-1]#=df_workload_day/480


# In[157]:


st.write('Statistiche descrittive')
st.write(df_fmr_day.describe().T)


# In[150]:


#df_fmr_day


# In[163]:


for j in colonne_tempi_ciclo:
    fig = px.line(df_fmr_day,  y=j, title='Run chart FMR',
                 template='plotly_dark')
    fig.update_traces(line_color='red')
    fig.show()
    st.plotly_chart(fig, theme=None, use_container_width=True)


# In[152]:


#st.plotly_chart(fig, theme=None, use_container_width=True)


# In[ ]:




