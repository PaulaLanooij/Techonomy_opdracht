# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 20:22:11 2024

@author: Paula
"""
import requests
import numpy as np
import copy 
import dutch_words
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

#%% Boeken verkrijgen op basis van woord in titel 

def fetch_all_books(query):
    api_url = "https://www.googleapis.com/books/v1/volumes" #Linkje
    params = {'q': f'intitle:{query}', 'maxResults': 40} #Parameters meegeven waarop je gaat zoeken

    all_books = [] #Leeg lijstje met boeken info

    while True:
        response = requests.get(api_url, params=params)

        if response.status_code == 200: #Als het openen van de API is goedgekeurd
            books_data = response.json().get('items', [])
            all_books.extend(books_data)

            # Kijk of er meer pagina's zijn om door te bladeren
            next_link = response.json().get('nextLink')
            if not next_link:
                break

            # Update 'startIndex' voor de volgende pagina
            params['startIndex'] = params.get('startIndex', 0) + params['maxResults']
        else:
            print(f"Error: {response.status_code}")
            return None

    return all_books

words = dutch_words.get_ranked()
woorden = words[0:100] #Pak de woorden 0 t/m 100 want anders wordt het te veel


df = pd.DataFrame()

for woord in woorden: #Zoek per woord op boeken die dat woord in de titel/subtitel hebben en voeg hiervan de info toe aan een dataframe. Zo maak je je eigen dataset. 
    books_data = fetch_all_books(woord)
    books_list = []
    for i in range(len(books_data)):
        volume_info = books_data[i].get('volumeInfo', {})
        book_dict = {
            'Titel': volume_info.get('title', 'N/A'),
            'Auteurs': ', '.join(volume_info.get('authors', ['N/A'])),
            'Publicatiejaar': volume_info.get('publishedDate', 'N/A'),
            'Taal': volume_info.get('language', 'N/A'),
            'Gemiddelde rating': volume_info.get('averageRating', 'N/A'),
            'Aantal paginas': volume_info.get('pageCount', 'N/A'),
            'Categorie': ', '.join(volume_info.get('categories', ['N/A']))}
        books_list.append(book_dict)
        df_books = pd.DataFrame(books_list)
        df_books.replace('N/A', np.nan, inplace = True) #Verwijder de rijen met NAN, we willen namelijk de rating voorspellen en die moet wel een gegeven zijn
        df_books = df_books.dropna()
        df = pd.concat([df, df_books], ignore_index=True)

#%% Veranderen van de data/klaarmaken voor het model

df = df[~df.apply(lambda row: row.astype(str).str.contains('\?').any(), axis=1)]
df['Publicatiejaar'] = df['Publicatiejaar'].astype(str).str[:4].astype(int)

df_dcat = pd.get_dummies(df['Categorie'], prefix='Cat')
df_dtaal = pd.get_dummies(df['Taal'], prefix = 'Taal')
df.drop(['Categorie', 'Taal'], axis=1, inplace=True)
                          
# Voeg de dummyvariabelen toe aan het oorspronkelijke DataFrame
df = pd.concat([df, df_dcat, df_dtaal], axis=1)

#Kopie maken dataframe en verwijderen titel en auteut
df_origineel = copy.deepcopy(df)
df.drop(['Titel','Auteurs'], axis = 1, inplace = True)

#%% Prepareren van de data voor het machine learning classificatie model

# Voorbeeld: Annoteer de ratings met categorieën
# Hier gebruik ik willekeurige grenzen; je moet de grenzen aanpassen op basis van je specifieke doelstellingen
df['Gemiddelde rating'] = pd.cut(df['Gemiddelde rating'], bins=[0, 2, 4, 5], labels=['Slecht', 'Gemiddeld', 'Goed'])

# Kenmerken en doelvariabele selecteren
Y = df['Gemiddelde rating']
X = df.drop('Gemiddelde rating', axis = 1)

#%% Max depth bepalen

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

#Om overfitting te voorkomen gaan we de maximale diepte van de boom analyseren. We gaan dus per diepte kijken naar de waarden van de accuracy bij de test en de train data 
max_depth_waarden = range(1,21)

train_accuracy = []
test_accuracy = []

for max_depth in max_depth_waarden:
    # Maak een Decision Tree Classifier
    clf = DecisionTreeClassifier(max_depth = max_depth, random_state=2)
    
    # Train het model op de trainingsgegevens
    clf.fit(X_train, y_train)
    
    # Accuracy als je naar de trainingsdata kijkt
    y_pred_test = clf.predict(X_train)
    train_acc = accuracy_score(y_pred_test,y_train)
    train_accuracy.append(train_acc)
    
    # Accuracy als je naar de testdata kijkt
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test,y_pred)
    test_accuracy.append(test_acc)


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(max_depth_waarden, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(max_depth_waarden, test_accuracy, label='Testing Accuracy', marker='o')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Classifier: Max Depth vs Accuracy')
plt.legend()
plt.grid(True)
plt.show()

#%% Met de gekozen max_depth doorgaan. Ik heb gekozen voor een max_depth van 6. Aangezien vervolgens een complexer model niet zorgt voor een significante verbetering.

clf = DecisionTreeClassifier(max_depth = 6, random_state=2)
clf.fit(X_train, y_train)

# Plot de beslissingsboom
plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=np.unique(Y), rounded=True)
plt.show()

y_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test,y_pred)
    
# Evaluatie van het model
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Toon de resultaten
print(f"Accuracy: {test_acc:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

#%% Visuals confusion matrix

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Voorspelde Labels')
plt.ylabel('Werkelijke Labels')
plt.title('Confusion Matrix')
plt.show()

#%% Visuals feature importance

import matplotlib.pyplot as plt

# Voorbeeld data
feature_importance = clf.feature_importances_
feature_names = X_train.columns

# Maak een staafdiagram voor de feature importance
plt.figure(figsize=(10, 18))  # Pas de figsize aan om de grootte van de plot te wijzigen
plt.barh(feature_names, feature_importance)
plt.xlabel('Importance')
plt.ylabel('Kenmerken')
plt.title('Feature Importance')
plt.show()

#%% Hoef je maar een keer te runnen

import nest_asyncio
nest_asyncio.apply()

#%% Data preparen voor voorspelling 

def data_preparation(df,publicatiejaar,taal,aantal_paginas,categorie):
    data = pd.DataFrame(columns = df.columns)
    data.loc[0] = [0] * len(data.columns)
    #data = data.append(pd.Series(0, index=df.columns), ignore_index=True)
    data.drop('Gemiddelde rating', axis = 1, inplace = True)
    data['Publicatiejaar'] = publicatiejaar
    data['Taal_' + taal] = 1 # nl, en
    data['Aantal paginas'] = aantal_paginas
    data['Cat_' + categorie] = 1 # 'Cat_American drama', 'Cat_Amsterdam (Netherlands)', 'Cat_Architects',
       #'Cat_Biography & Autobiography', 'Cat_Body, Mind & Spirit',
       #'Cat_Buddhism', 'Cat_Business & Economics',
       #'Cat_Cabinets of curiosities', 'Cat_Christmas stories', 'Cat_Cooking',
       #'Cat_Crafts & Hobbies', 'Cat_Cultural landscapes', 'Cat_Design',
       #'Cat_Dutch fiction', 'Cat_Education', 'Cat_Evolution (Biology)',
       #'Cat_Family & Relationships', 'Cat_Fiction', 'Cat_Four-color problem',
       #'Cat_Geodynamics', 'Cat_Gouda (Netherlands)', 'Cat_Governors',
       #'Cat_History', 'Cat_Holocaust, Jewish (1939-1945)',
       #'Cat_Juvenile Fiction', 'Cat_Juvenile Nonfiction',
       #'Cat_Literary Criticism', 'Cat_Medical', 'Cat_Philosophy',
       #'Cat_Psychology', 'Cat_Religion', 'Cat_Religious drama, Dutch',
       #'Cat_Science', 'Cat_Technology & Engineering', 'Cat_Templars',
       #'Cat_True Crime', 'Cat_Young Adult Fiction'     
    return data


#%% FastAPI

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn


class Item(BaseModel):
    Publicatiejaar: int = Field(..., description="Publicatiejaar van het item in getallen")
    Taal: str = Field(..., description="nl of en")
    Aantal_paginas: int = Field(..., description = "Aantal paginas in getallen")
    Categorie: str = Field(..., description = "Keuze uit: American drama, Amsterdam (Netherlands), Architects, Biography & Autobiography, 'Body, Mind & Spirit', Buddhism, Business & Economics,"
    "Cabinets of curiosities, Christmas stories, Cooking, Crafts & Hobbies, Cultural landscapes, Design, Dutch fiction, Education, Evolution (Biology), Family & Relationships, Fiction, Four-color problem,"
    "Geodynamics, Gouda (Netherlands), Governors, History, Holocaust, Jewish (1939-1945), Juvenile Fiction, Juvenile Nonfiction, Literary Criticism, Medical, Philosophy, Psychology, Religion, Religious drama, Dutch,"
    "Science, Technology & Engineering, Templars, True Crime, Young Adult Fiction")
    
app = FastAPI()    
    

@app.get("/")
async def root():
    return {"message": "Hello, this is my first FastAPI app!"}


@app.post("/opslaan_item")
def opslaan_item(item: Item):
    input_data = {
        "publicatiejaar": item.Publicatiejaar,
        "taal": item.Taal,
        "aantal_paginas": item.Aantal_paginas,
        "categorie": item.Categorie
    }
    return input_data

@app.post("/predict_rating")
def predict_rating(item: Item):
    publicatiejaar = item.Publicatiejaar
    taal = item.Taal
    aantal_paginas = item.Aantal_paginas
    categorie = item.Categorie
    
    input_data = data_preparation(df, publicatiejaar, taal, aantal_paginas, categorie) 
    print(input_data)
    
    predicted_rating = clf.predict(input_data[0:])
    print(predicted_rating)

    return {"predicted_rating": predicted_rating}

if __name__ == '__main__':
    uvicorn.run(app)
# %%
