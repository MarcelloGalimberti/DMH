#!/usr/bin/env python
# coding: utf-8

# In[193]:


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


# #### Importo tabelle da GitHub

# In[93]:


# Creare una funzione che carica i dati da url + indice e decorarla con @st.cache_data


# In[94]:


@st.cache_data
def load_data(url, index=''):
    if index:
        data = pd.read_excel(url, index_col=index)
    else:
        data = pd.read_excel(url)

    return data


# In[95]:


url_tempi_opt = 'https://github.com/MarcelloGalimberti/DMH/blob/main/db_tempi_opt_reale.xlsx?raw=true'


# In[96]:


df_tempi_opt = load_data(url_tempi_opt,index = 'Bundle')


# In[97]:


url_take_rate = 'https://github.com/MarcelloGalimberti/DMH/blob/main/db_take_rate.xlsx?raw=true'


# In[98]:


df_take_rate = load_data(url_take_rate)


# ### Deve diventare input

# In[99]:


url_vincoli = 'https://github.com/MarcelloGalimberti/DMH/blob/main/Vincoli_produzione.xlsx?raw=true'


# In[100]:


# vincoli_produzione = pd.read_excel(url_vincoli,index_col='Origine')


# ---

# In[101]:


url_tempi_base = 'https://github.com/MarcelloGalimberti/DMH/blob/main/db_tempi_base.xlsx?raw=true'


# In[102]:


tempi_base = load_data(url_tempi_base,index='Versione')


# In[103]:


url_immagine = 'https://github.com/MarcelloGalimberti/DMH/blob/main/Ducati-Multistrada-V4-2021-008.jpg?raw=true'


# In[104]:


# immagine_locale = '/Users/marcello_galimberti/Downloads/Ducati-Multistrada-V4-2021-008.jpg'


# ---

# #### Grafico sunburst del take rate

# In[105]:


# mostrare percentuali


# In[106]:


fig = px.sunburst(df_take_rate,
                 path=['Versione','Optional'],
                 values = '%_Optional',
                 color='%_Optional',
                 color_continuous_scale="reds",
                 title = 'Take rate',
                 width=850, height=750)
fig.update_layout(paper_bgcolor="rgb(0,0,0,0)",
                 font_color="white")


# #### Titolo

# In[107]:


# CREARE UNA FUNZIONE PER CARICARE L'IMMAGINE E POI DECORARLA


# In[108]:


st.image(url_immagine)


# In[109]:


#display(Image.open(immagine_locale))


# In[110]:


#display(Image.open(url_immagine))


# In[111]:


st.title('MTSV4 MTO Factory Feasibility')
st.write('Database delta tempi bundle')
st.write(df_tempi_opt)


# In[112]:


st.plotly_chart(fig, theme=None, use_container_width=True)


# #### Test grafico con slider

# In[113]:


#stgr_installati = st.slider('stgr installati',0,50,(25))
#linea_installati = st.slider('linea installati',0,50,(43))


# In[114]:


#data = {
#    'Area_Produzione': ['stgr', 'linea'],
#    'Risorse': [stgr_installati, linea_installati]
#}
#df = pd.DataFrame(data)


# In[115]:


# Set the Seaborn style and color palette
#sns.set_style('dark')
#sns.set_palette(sns.color_palette("Reds"))


# In[116]:


#fig, ax = plt.subplots()
#sns.barplot(x='Area_Produzione', y='Risorse', data=df, ax=ax)
#ax.set_xlabel('Area Produzione')
#ax.set_ylabel('Risorse')
#ax.set_title('Risorse installate')

# Add labels to the bars
#for p in ax.patches:
#    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
#                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points', color='white')


# In[117]:


#st.pyplot(fig,use_container_width=True)


# ---

# #### Orario di lavoro e veicoli/mese (input utente)

# In[118]:


# Giorni_Lavorativi = 21.9
# Ore_giorno = 8


# In[119]:


Giorni_Lavorativi = st.number_input('Giorni lavorativi nel mese')
st.write('I giorni lavorativi sono: ', Giorni_Lavorativi)


# In[120]:


Ore_giorno = st.number_input('Ore / giorno')
st.write('Le ore / giorno sono: ', Ore_giorno)


# In[121]:


#Giorni_Lavorativi = 21.9
#Ore_giorno = 8


# In[122]:


Tempo_disponibile = Giorni_Lavorativi*Ore_giorno*60 # uguale a Takt Time (1 moto)


# In[123]:


st.write('Il tempo disponibile nel mese è di [min] ',Tempo_disponibile)


# In[124]:


Veicoli_mese = st.number_input('Veicoli / mese')
st.write('I veicoli / mese sono: ', Veicoli_mese)


# In[125]:


#Veicoli_mese=1482


# ---

# ## Processo dati

# #### Creo database take rate con volumi mese

# In[126]:


#@st.cache_data
def take_rate (Veicoli_mese):
    df_take_rate['Qty versione']=(df_take_rate['%_Versione']*Veicoli_mese).round().astype(int)
    df_take_rate['Qty optional']=(df_take_rate['%_Optional']*df_take_rate['Qty versione']).round().astype(int)
    return df_take_rate


# In[127]:


take_rate(Veicoli_mese) # dato da input in Streamlit altrimenti 1482


# #### Creo database take rate per ogni versione

# In[128]:


df_PP = df_take_rate[(df_take_rate['Versione'] == 'Pikes Peak')]
df_Rally = df_take_rate[(df_take_rate['Versione'] == 'Rally')]
df_RS = df_take_rate[(df_take_rate['Versione'] == 'RS')]
df_S = df_take_rate[(df_take_rate['Versione'] == 'S')]
df_S_Grand_Tour = df_take_rate[(df_take_rate['Versione'] == 'S Grand Tour')]
df_Standard = df_take_rate[(df_take_rate['Versione'] == 'Standard')]


# #### Funzione che crea df vuoto in base alla versione (es: arg = df_PP)

# In[129]:


#@st.cache_data
def crea_df_veicoli (df): # inserire ad esempio df_PP
    lista_optional = list(df['Optional'].unique())
    df_veicoli_creato = pd.DataFrame(columns =  lista_optional, index = range(df['Qty versione'].max()))
    return df_veicoli_creato


# In[130]:


df_veicoli_PP = crea_df_veicoli(df_PP)
df_veicoli_Rally = crea_df_veicoli(df_Rally)
df_veicoli_RS = crea_df_veicoli(df_RS)
df_veicoli_S = crea_df_veicoli(df_S)
df_veicoli_S_Grand_Tour = crea_df_veicoli(df_S_Grand_Tour)
df_veicoli_Standard = crea_df_veicoli(df_Standard)


# #### Funzione che compila il database vuoto con il numero di opt per veicolo
# es: arg (df_PP, df_veicoli_PP)

# In[131]:


#@st.cache_data
def compila_veicoli (df,df_veicoli):
    for i in range (len(df)):
        for j in range (len(df_veicoli)):
            if j < df.iloc[i]['Qty optional']:
                df_veicoli.iat[j,i] = 1
            else:
                df_veicoli.iat[j,i] = 0
    df_veicoli.index.name='id_veicolo'
    return df_veicoli


# In[132]:


df_veicoli_PP = compila_veicoli (df_PP, df_veicoli_PP)
df_veicoli_Rally = compila_veicoli (df_Rally,df_veicoli_Rally)
df_veicoli_RS = compila_veicoli (df_RS,df_veicoli_RS)
df_veicoli_S = compila_veicoli (df_S,df_veicoli_S)
df_veicoli_S_Grand_Tour = compila_veicoli (df_S_Grand_Tour,df_veicoli_S_Grand_Tour)
df_veicoli_Standard = compila_veicoli (df_Standard,df_veicoli_Standard)


# #### Funzione che fa shuffle per ogni colonna del df_veicoli con opt

# In[133]:


#@st.cache_data
def shuffle_veicoli (df_veicoli):
    for i in df_veicoli.columns:
        df_veicoli[i]=df_veicoli[i].sample(frac=1).values
    return df_veicoli


# In[134]:


df_veicoli_PP = shuffle_veicoli(df_veicoli_PP)
df_veicoli_Rally = shuffle_veicoli(df_veicoli_Rally)
df_veicoli_RS = shuffle_veicoli(df_veicoli_RS)
df_veicoli_S = shuffle_veicoli(df_veicoli_S)
df_veicoli_S_Grand_Tour = shuffle_veicoli(df_veicoli_S_Grand_Tour)
df_veicoli_Standard = shuffle_veicoli(df_veicoli_Standard)


# #### Funzione che carica i tempi dei veicoli con opt
# es: arg(df_veicoli_PP, df_delta_t)
# ** LUNGO RUNTIME **

# In[135]:


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


# In[136]:


df_delta_tempi_PP = carica_delta_t(df_veicoli_PP)
df_delta_tempi_Rally = carica_delta_t(df_veicoli_Rally)
df_delta_tempi_RS = carica_delta_t(df_veicoli_RS)
df_delta_tempi_S = carica_delta_t(df_veicoli_S)
df_delta_tempi_S_Grand_Tour = carica_delta_t(df_veicoli_S_Grand_Tour)
df_delta_tempi_Standard = carica_delta_t(df_veicoli_Standard)


# In[137]:


df_delta_tempi_PP['Versione']='Pikes Peak'
df_delta_tempi_Rally['Versione'] = 'Rally'
df_delta_tempi_RS['Versione'] = 'RS'
df_delta_tempi_S['Versione'] = 'S'
df_delta_tempi_S_Grand_Tour['Versione'] = 'S Grand Tour'
df_delta_tempi_Standard['Versione'] = 'Standard'


# In[138]:


df_delta_tempi_PP.set_index('Versione',inplace=True)
df_delta_tempi_Rally.set_index('Versione',inplace=True)
df_delta_tempi_RS.set_index('Versione',inplace=True)
df_delta_tempi_S.set_index('Versione',inplace=True)
df_delta_tempi_S_Grand_Tour.set_index('Versione',inplace=True)
df_delta_tempi_Standard.set_index('Versione',inplace=True)


# In[139]:


df_delta_tempi_PP = df_delta_tempi_PP.add(tempi_base)
df_delta_tempi_Rally = df_delta_tempi_Rally.add(tempi_base)
df_delta_tempi_RS = df_delta_tempi_RS.add(tempi_base)
df_delta_tempi_S = df_delta_tempi_S.add(tempi_base)
df_delta_tempi_S_Grand_Tour = df_delta_tempi_S_Grand_Tour.add(tempi_base)
df_delta_tempi_Standard = df_delta_tempi_Standard.add(tempi_base)


# In[140]:


df_delta_tempi_PP.dropna(inplace = True)
df_delta_tempi_Rally.dropna(inplace = True)
df_delta_tempi_RS.dropna(inplace = True)
df_delta_tempi_S.dropna(inplace = True)
df_delta_tempi_S_Grand_Tour.dropna(inplace = True)
df_delta_tempi_Standard.dropna(inplace = True)


# In[141]:


# QUI RESETTARE INDICE


# In[142]:


df_delta_tempi_PP.reset_index(inplace=True)
df_delta_tempi_Rally.reset_index(inplace=True)
df_delta_tempi_RS.reset_index(inplace=True)
df_delta_tempi_S.reset_index(inplace=True)
df_delta_tempi_S_Grand_Tour.reset_index(inplace=True)
df_delta_tempi_Standard.reset_index(inplace=True)


# In[143]:


# vertical_concat serve?


# In[144]:


vertical_concat = pd.concat([df_delta_tempi_PP,
                             df_delta_tempi_Rally,
                            df_delta_tempi_RS,
                            df_delta_tempi_S,
                            df_delta_tempi_S_Grand_Tour,
                            df_delta_tempi_Standard], axis=0)


# #### Unisce database veicoli

# In[145]:


df_delta_t_opt_PP = pd.concat([df_delta_tempi_PP,df_veicoli_PP], axis=1)
df_delta_t_opt_Rally = pd.concat([df_delta_tempi_Rally,df_veicoli_Rally], axis=1)
df_delta_t_opt_RS = pd.concat([df_delta_tempi_RS,df_veicoli_RS], axis=1)
df_delta_t_opt_S = pd.concat([df_delta_tempi_S,df_veicoli_S], axis=1)
df_delta_t_opt_S_Grand_Tour = pd.concat([df_delta_tempi_S_Grand_Tour,df_veicoli_S_Grand_Tour], axis=1)                              
df_delta_t_opt_Standard = pd.concat([df_delta_tempi_Standard,df_veicoli_Standard], axis=1)


# In[146]:


unione_veicoli = pd.concat([df_delta_t_opt_PP,
                            df_delta_t_opt_Rally,
                            df_delta_t_opt_RS,
                            df_delta_t_opt_S,
                            df_delta_t_opt_S_Grand_Tour,
                            df_delta_t_opt_Standard,
                           ], ignore_index= True,axis = 0).fillna(0).set_index('Versione')


# In[147]:


unione_veicoli_caratteristiche = unione_veicoli.copy()


# In[148]:


# QUI VISULIZZARE ED ESPORTARE


# In[149]:


unione_veicoli.drop(unione_veicoli.columns[5:21], inplace=True, axis=1)


# In[150]:


# unione_veicoli #altrinenti viene visualizzato in streamlit


# #### Crea fabbisogno risorse

# In[151]:


#Giorni_Lavorativi = 21.9
#Ore_giorno = 8


# In[152]:


#Tempo_disponibile = Giorni_Lavorativi*Ore_giorno*60


# In[153]:


fabbisogno_risorse = unione_veicoli.copy()


# In[154]:


fabbisogno_risorse = fabbisogno_risorse/Tempo_disponibile


# In[155]:


df_fabbisogno = pd.DataFrame(fabbisogno_risorse.sum()).T


# In[156]:


df_fabbisogno['Origine']='Fabbisogno'


# In[157]:


df_fabbisogno.set_index(['Origine'], inplace=True)


# In[158]:


st.write('Fabbisogno risorse')
st.write(df_fabbisogno)


# In[159]:


df_fabbisogno # da mettere come output fabbisogno FMR


# #### Confronto tra fabbisogno e risorse disponibili

# In[161]:


# creare dataframe vincoli_produzione con slider
# 


# In[162]:


st.write('Inserire capacità installata')


# In[163]:


stgr_installati = st.slider('stgr installati',1,50,(25))
linea_installati = st.slider('linea installati',1,50,(43))
collaudo_installati = st.slider('collaudo installati',1,20,(12))
radar_installati = st.slider('radar installati',1,2,(1))
vestizione_installati = st.slider('vestizione installati',1,20,(20))


# In[164]:


#stgr_installati = 25
#linea_installati = 43
#collaudo_installati = 12
#radar_installati = 1
#vestizione_installati = 20


# In[165]:


colonne_v_p = ['Origine', 'stgr','linea','collaudo','radar','vestizione']


# In[166]:


numero_risorse = ['Capacità installata',
                  stgr_installati,
                  linea_installati,
                  collaudo_installati,
                  radar_installati,
                  vestizione_installati]


# In[167]:


df_test = pd.DataFrame(columns=colonne_v_p)


# In[168]:


df_test.loc[len(df_test)] = numero_risorse


# In[169]:


df_test.set_index('Origine',inplace=True)


# In[170]:


vincoli_produzione = df_test.copy()


# In[171]:


df_confronto = pd.concat([vincoli_produzione,df_fabbisogno],axis=0)


# In[172]:


st.write('Fabbisogno vs Capacità')
st.write(df_confronto)


# In[173]:


df_confronto # VISUALIZZARE?


# #### Crea saturazione

# In[174]:


saturazione = []
for i in range(len(df_confronto.columns)):
    sat = (df_confronto.iloc[1][i]/df_confronto.iloc[0][i])
    saturazione.append(sat)


# In[175]:


lista_colonne_saturazione = list(df_confronto.columns)


# In[176]:


df_saturazione = pd.DataFrame(saturazione)


# In[177]:


df_saturazione=df_saturazione.T


# In[178]:


df_saturazione.columns = lista_colonne_saturazione


# In[179]:


# df_saturazione # poi si mostra come grafico


# #### Bar chart saturazione

# In[180]:


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

# #### Crea id veicolo

# In[181]:


unione_veicoli_caratteristiche.reset_index(inplace=True)


# In[182]:


unione_veicoli_caratteristiche['C'] = unione_veicoli_caratteristiche.groupby('Versione').cumcount()+1


# In[183]:


unione_veicoli_caratteristiche['C']=unione_veicoli_caratteristiche['C'].astype(str)


# In[184]:


colonne_id = ['Versione','C']
unione_veicoli_caratteristiche['id']=unione_veicoli_caratteristiche[colonne_id].apply(lambda x: '|'.join(x), axis=1)


# In[185]:


colonne_tempi_ciclo = ['stgr','linea','collaudo','radar','vestizione']


# In[186]:


#unione_veicoli_caratteristiche


# In[187]:


unione_veicoli_caratteristiche['TCT']=unione_veicoli_caratteristiche[colonne_tempi_ciclo].sum(axis=1)


# In[188]:


#unione_veicoli_caratteristiche


# In[189]:


unione_veicoli_caratteristiche['CV']=unione_veicoli_caratteristiche[colonne_tempi_ciclo].std(axis=1)/unione_veicoli_caratteristiche[colonne_tempi_ciclo].mean(axis=1)


# In[190]:


unione_veicoli_caratteristiche


# In[211]:


#unione_veicoli_caratteristiche.to_excel('/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Ducati 2023/ASC/Simulazione/unione_veicoli_caratteristiche_CV.xlsx')


# #### Creo 2 cluster

# In[191]:


X = unione_veicoli_caratteristiche.loc[:,['TCT','CV']] # tempi ciclo fasi principali


# In[192]:


#X


# In[194]:


scaler = StandardScaler()


# In[195]:


# fit and transform the data
scaled_data = scaler.fit_transform(X) 


# In[196]:


#scaled_data


# In[197]:


# definisco istanza di KMeans
km = KMeans(n_clusters = 2,
           init ='k-means++',
           n_init=10,
           max_iter=300,
           tol=1e-04,
           random_state=0)


# In[198]:


clusters=km.fit_predict(scaled_data)


# In[209]:


df_describe = pd.DataFrame(clusters)
#df_describe.value_counts()


# In[210]:


#df_describe


# In[212]:


#df_describe.to_excel('/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Ducati 2023/ASC/Simulazione/df_describe.xlsx')


# In[213]:


unione_veicoli_caratteristiche=pd.concat([unione_veicoli_caratteristiche,df_describe],axis=1)


# In[218]:


unione_veicoli_caratteristiche.rename(columns = {0:'Cluster'}, inplace = True)


# In[219]:


#unione_veicoli_caratteristiche.columns


# #### Assegna criticità

# In[ ]:


# possibile input: percentile


# In[ ]:


#unione_veicoli_caratteristiche['stgr_critico']=(unione_veicoli_caratteristiche['stgr']>=unione_veicoli_caratteristiche['stgr'].quantile(0.75))
#unione_veicoli_caratteristiche['linea_critico']=(unione_veicoli_caratteristiche['linea']>=unione_veicoli_caratteristiche['linea'].quantile(0.75))
#unione_veicoli_caratteristiche['collaudo_critico']=(unione_veicoli_caratteristiche['collaudo']>=unione_veicoli_caratteristiche['collaudo'].quantile(0.75))
#unione_veicoli_caratteristiche['radar_critico']=(unione_veicoli_caratteristiche['radar']>=unione_veicoli_caratteristiche['radar'].quantile(0.75))
#unione_veicoli_caratteristiche['vestizione_critico']=(unione_veicoli_caratteristiche['vestizione']>=unione_veicoli_caratteristiche['vestizione'].quantile(0.75))


# In[220]:


#unione_veicoli_caratteristiche['critico']=unione_veicoli_caratteristiche['TCT']>=unione_veicoli_caratteristiche['TCT'].quantile(0.75)
unione_veicoli_caratteristiche['critico']=unione_veicoli_caratteristiche['Cluster']==0


# ####  Processo di decisione su gruppo risorse critiche

# In[224]:


#critica=df_saturazione.idxmax(axis=1).astype(str)+'_critico'


# In[225]:


#risorsa_critica=critica[0]


# In[226]:


#risorsa_critica # valutare come dare info all'utente


# #### Processo selezione liste heijunka

# In[227]:


lista_heihunka = []
lista_nc = []
lista_c = []


# In[228]:


for i in range (len(unione_veicoli_caratteristiche)):
    if unione_veicoli_caratteristiche.iloc[i]['critico']==True:
        lista_c.append(unione_veicoli_caratteristiche.loc[i]['id'])
    else:
        lista_nc.append(unione_veicoli_caratteristiche.loc[i]['id'])


# In[229]:


len(lista_nc)


# In[230]:


len(lista_c)


# In[231]:


lista_heihunka=[lista_c,lista_nc]


# #### Algoritmo heijunka

# In[232]:


count = sum([ len(listElem) for listElem in lista_heihunka])


# In[233]:


sequenza = list(range(count))
residuo = list(range(count))
for i in range(len(lista_heihunka)):
    if i == 0:
        indice_versione = []
        indice_versione = np.linspace(0,len(residuo)-1,len(lista_heihunka[i]),dtype=int)
        counter = 0
    #residuo = indice_versione
        for k in indice_versione:
            sequenza[k]=lista_heihunka[i][counter]
            counter = counter +1
    else:
        residuo = np.delete(residuo,indice_versione)
        counter = 0
        #indice_versione_1 = np.linspace(0,len(residuo)-1,len(lista_heihunka[i]),dtype=int)
        for k in residuo:
            sequenza[k]=lista_heihunka[i][counter]
            counter = counter +1


# In[234]:


d = {'Sequenza': sequenza}
df_heijunka = pd.DataFrame(d)


# In[235]:


df_heijunka_ind = df_heijunka.copy()


# In[236]:


df_heijunka_ind['id']=df_heijunka_ind['Sequenza']


# In[237]:


df_heijunka_ind.set_index('id',inplace=True)


# In[238]:


df_uvc = unione_veicoli_caratteristiche.copy()


# In[239]:


df_uvc.set_index('id',inplace=True)


# In[240]:


df_sequenza = pd.concat([df_heijunka_ind,df_uvc],axis = 1)


# #### Visualizzare sequenza

# In[241]:


#df_sequenza


# In[242]:


#df_sequenza.to_excel('/Users/marcello_galimberti/Documents/Marcello Galimberti/ADI business consulting/Offerte 2023/Ducati 2023/ASC/Simulazione/df_sequenza_cluster.xlsx')


# In[ ]:


st.dataframe(df_sequenza)


# In[ ]:


# pare che funzioni
# devo esportare i requirements

