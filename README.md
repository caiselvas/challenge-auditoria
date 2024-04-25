# Challenge-auditoria (WINNERS of the Challenge)

> Aquest repositori conté el model desenvolupat i creat per l'equip GIA UPC (Roger Baiges Trilla, Pau Hidalgo Pujol i Cai Selvas Sala) per al challenge del col·legi d'auditors Censors de Catalunya d'auditoria i IA.

## Taula de Continguts

- [Challenge-auditoria](#challenge-auditoria)
  - [Taula de Continguts](#taula-de-continguts)
  - [Descripció](#descripció)
  - [Estructura dels fitxers](#estructura-dels-fitxers)
  - [Requeriments](#requeriments)
  - [Ús](#ús)
  - [Problemes](#problemes)

## Descripció

Aquest projecte conté la implementació del model proposat per al challenge. Consisteix en un preprocessament inicial del fitxer .xlsx, per tal d'ajuntar les dades en un de sol i crear noves variables.

A partir d'aquí, es planteja el càlcul d'un "índex" relacionat amb el deteriorament usant tres mètodes diferents:
- Una fòrmula que pondera diversos factors, explicada al document i a la presentació
- Un valor basat en la predicció realitzada amb un model de sèries temporals (ARIMA, nbeats, timesnet)...
- Finalment, un basat en un autoencoder, que aprofita el seu espai latent per calcular la similitud amb un element molt negatiu. Com més a prop, més "deteriorament" se suposa que tindrà el producte
  
Aquests tres valors es combinen per realitzar una predicció final. Aquesta, a més, és analitzada per un EBM per tal d'establir quines han sigut les variables que han influït més i de quina forma.

Com a extra, també hi ha la implementació de dos models més: un simple detector d'outliers, basat en Isolation Forest, a la carpeta outliers, i un mini-Xat, a la carpeta xat_rag, que utilitza tècniques de NLP (processament del llenguatge natural) i de RAG (retrieval augmented generation) per tal de realitzar cerques a la base de dades (o fins i tot algun càlcul) en base al que li diu l'usuari. Mencionar que aquest últim model és bastant bàsic (té poques funcionalitats), però evidentment té molt marge de millora.

## Estructura dels fitxers

L'estructura dels fitxers d'aquest directori és:


    ├── run_model.ipynb                   # Notebook per tal d'executar el model principal
    ├── create_new_dataset.py             # Fitxer Python per crear el nou dataset "preprocessat"
    ├── inventory_impairment_class.py     # Implementació del classe del model 
    ├── xat_rag                    # Carpeta que conté els fitxers del xat
    |   ├── xat_rag.ipynb        # Notebook per crear i interactuar amb el Xat
    │   ├── queries.pkl          # Base de dades del xat
    │   └── universal_sentence_encoder.tflite     # Model d'embeddings d'internet usat pel xat
    ├── outliers                   # Carpeta que conté els fitxers de detecció d'outliers
    │   └── isolation_forest_example.ipynb  # Exemple de detecció d'outliers amb isolation forest 
    ├── monthly_ts                    # Carpeta que conté els fitxers de les sèries temporals
    |   ├── neural_forecast.ipynb        # Notebook per crear les prediccions amb models neuronals
    │   ├── arima_forecast.ipynb           # Notebook per crear les prediccions amb ARIMA (implementat també en el model directament)
    │   └── add monthly_data.py     # Creació de les dades mensuals fictícies  (implementat també en el model directament)
    ├── forecast                    # Carpeta que conté les prediccions de les sèries temporals
    |   ├── arima.json        # JSON que conté les prediccions pels pròxims 12 mesos usant ARIMA
    |   ├── FEDformer.json        # JSON que conté les prediccions pels pròxims 12 mesos usant FEDformer
    |   ├── nbeats.json        # JSON que conté les prediccions pels pròxims 12 mesos usant NBEATS
    │   └── timesnet.json     # JSON que conté les prediccions pels pròxims 12 mesos usant TimesNet
    ├── data                    # Carpeta que els diferents Excels en estats diferents
    │   └── results_inventory_impairment.xlsx    # Excel final creat pel model
    ├── reports                    # Informes de descripció del projecte
    └── README.md

## Requeriments

Cal tenir instal·lat Python. Recomanem usar VSCode per tal de poder treballar còmodament amb els notebooks.

Les llibreries necessàries es poden instal·lar fent:
```
pip install -r requirements.txt
```


Per tal d'utilitzar el Xat, tenir en compte que NO fuciona en ordinadors Mac.

## Ús

Utilitzar el model és molt senzill. Tan sols cal entrar al notebook `run_model.ipynb`, i executar tot el codi (a VSCode, a dalt hi ha n botó de RunAll). Per fer aquest procés més senzill, a data ja hi ha tots els documents necessaris. En cas que hi hagués algun problema, podria ser que fes falta executar create new dataset per crear el fitxer que usa el model.

Per executar `xat_rag.ipynb`, cal tenir el fitxer `/data/results_inventory_impairment.xlsx`, que és el que crea el model.

## Problemes

Si teniu qualsevol problema, no dubteu en crear una nova issue. Intentarem resoldre'l, si podem.
