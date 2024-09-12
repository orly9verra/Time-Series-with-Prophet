
# Time-Series-with-Prophet
STOCHASTIC MODEL :TIME SERIES WITH PROPHET

# ETUDE PORTANT SUR LA SOCIETE NOZ 

# Problématique : 
NOZ est une société de vente au détail renommée qui exploite une chaîne d'hypermarchés. Ici, NOZ a fourni une combinaison de données de 45 magasins, y compris des informations sur les magasins et les ventes mensuelles. NOZ essaie d’abord de trouver l'impact des vacances sur les ventes de magasin pour lequel il a inclus quatre semaines de vacances dans l'ensemble de données qui sont Noël, Thanksgiving, Super Bowl, Fête du travail. Ensuite prédire les ventes des prochaines vacances. Ici, nous devons analyser l'ensemble de données fourni. 

# Objectif du Projet : 
Notre objectif principal est d’étudier l’impact des vacances sur les ventes des magasins et de prédire les ventes du magasin pour les prochaines vacances. Comme dans le jeu de données, la taille de l’ensemble de données et les données temporelles sont données en tant que caractéristique, nous analysons donc si les ventes sont affectées par des facteurs basés sur le temps. Plus important encore, comment l’inclusion des vacances dans une semaine fait grimper les ventes en magasin? Faire une prévision des ventes des vacances prochaines. Nous exécutons ce projet en utilisant "Prophet".

# Réalisation du  travail
Nous téléchargeons ici les packages nécessaires pour la réalisation des tâches 

# PARTIE 1 : PACKAGES & REPERTOIRE DE TRAVAIL 
## Packages 
import pandas as pd                  # to create a DataFrame tableau 
import pandas_datareader.data as web # package that allows us to create a pandas DataFrame object by using various data sources from the internet.
import numpy as np                   # calcul scientifique 
import datetime as dt                # traiter les dates 
import matplotlib.pyplot as plt      # ajouter des éléments tels que des lignes, des images ou des textes aux axes d'un graphique
from matplotlib import style         # tracer et visualiser des données sous formes de graphiques.
style.use ('ggplot')                 # tracer des fonctions 
import os
import prophet                       # time series forecating 
import seaborn as sns                # Plot
## Environnement de travail 

Après avoir télécharger les packages, nous organisons notre travail en créant un environnement de travail propre. Identifier l'emplacement des bases de données, tracer un chemin et enfin définir le lieu de stockages des résultats. 
# Connaître l'emplacement de base
os.getcwd()  
# tracer le chemein ( Path)
path="C:/Users/dell/Desktop/M2IFM/S2M2/stochastic model/"
os.chdir(path)
path 

# Connaître le contenu de notre répertoire

os.listdir()

# PARTIE 2 : IMPORT DATASETS  
# load data
walfare= pd.read_csv("Walmare_data.csv",sep=";", decimal = ".",)
walfare.head(5)

## PARTIE 3 : Traitement des informations 

# Valeurs manquantes 
print(walfare.isnull().sum(axis=0))

# nombre de variables & traitement d'info
walfare.shape

# Le nombre de Store 
walfare["Store"].nunique()

# le nombre de département 
walfare["Dept"].nunique()

# ### Sur quelle période porte notre étude
walfare['Date'].head(5).append(walfare['Date'].tail(5)) # to see first and last 5 rows.

#  * Quelles sont les ventes par store et departement?

Table_storedepart = pd.pivot_table(walfare, index='Store', columns='Dept',
                                  values='Weekly_Sales', aggfunc=np.mean)
display(Table_storedepart)

##  Visualisation des données
### Représentation Graphique des Ventes hebdomadaires par store
 plt.figure(figsize=(14,6))
sns.barplot(x='Store', y='Weekly_Sales', data=walfare)
# Représentation graphique des Ventes hebdomadaires  en fonction des vacances 

plt.figure(figsize=(14,6))
sns.barplot(x='IsHoliday', y='Weekly_Sales', data=walfare)

# Représentation graphique des Ventes hebdomadaires  par mois
monthly_sales = pd.pivot_table(walfare, values = "Weekly_Sales", columns = "year", index = "month")
monthly_sales.plot()
plt.figure()

plt.figure(figsize=(14,6))
fig = sns.barplot(x='month', y='Weekly_Sales', data=walfare)

# PARTIE4 - CREATE TIME SERIES - Training a Prophet Time Series Model

frame = walfare.iloc[:,[2,3]]
frame.head(5)

frame.columns=['ds','y']
frame.head()

frame['ds'] = pd.to_datetime(frame['ds'])
frame.head()

frame.plot(x='ds', y='y', figsize=(18,6))

Nous divisons la base en base de test et d'entraînement (Train Test split)

train = frame.iloc[: len(frame)-365]
test = frame.iloc[len(frame)-365]

## Nous entammons la prédiction (Pour cela nous téléchrgeons le package nécessaire pour cela.

from prophet import Prophet 
m = Prophet (interval_width = 0.95)
training_run = m.fit(frame)

# PARTIE5 : Making Predictions and evaluating Performance
future = m.make_future_dataframe(periods = 20, freq='D')
future.head(3)

forecast = m.predict(future)
forecast.tail()

forecast[['ds', 'yhat','yhat_lower', 'yhat_upper']].tail()

plotl = m.plot(forecast,  uncertainty=True)

plot2 = m.plot_components(forecast)

lines = walfare['Store'].unique()
lines

wal = walfare.iloc[:,[0,2,3]]
wal.head(5) 

for stock_line in lines : 
    frame =walfare[walfare['Store']== stock_line].copy()
    print (frame.tail())


fit_models ={}
for stock_line in lines:
    frame =wal[wal['Store']== stock_line].copy()
    frame.drop('Store', axis=1, inplace=True)
    frame.columns = ['ds', 'y']
    
    m = Prophet (interval_width = 0.95)
    model = m.fit(frame)
    fit_models[stock_line] = m
# Forecasting 
forward = fit_models[1].make_future_dataframe(20)
forecast = fit_models[1].predict(forward)

### Plotting the forecasted components 

fig1 = m.plot_components(forecast) 

Nous pouvons tracer la tendance et la saisonnalité, composantes de la prévision comme suit :

# Evaluation du model 

Après analyse statistiques et implémentation du modèle, nous passons à l'évaluation du modèle. 

from statsmodels.tools.eval_measures import rmse
store 5, 3, 33, 43
 predictions= = forecast.iloc [-365:] ['yhat']















