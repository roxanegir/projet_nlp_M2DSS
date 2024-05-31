# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:53:17 2024

@author: Roxane Girault, Marie Tapia, Chafiaa Challal
"""

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import unicodedata
import string
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer
import re
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt

def scrape_page(url):
    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(5)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)
    page_source = driver.page_source
    driver.quit()

    soup = BeautifulSoup(page_source, 'html.parser')
    review_cards = soup.find_all('div', class_='review-card')

    comments = []
    ratings = []
    commenters = []
    dates = []

    for card in review_cards:
        try:
            profile = card.find('div', class_='profile')
            commenter = profile.find('h4').text.strip()
            date = profile.find('span').text.strip()

            comment_section = card.find('div', class_='comment')
            title = comment_section.find('h5').text.strip()

            positive_comment = card.find('svg', {'id': 'smiling-face'})
            negative_comment = card.find('svg', {'class': 'feather feather-frown'})

            positive_text = positive_comment.find_next('p').text.strip() if positive_comment else ""
            negative_text = negative_comment.find_next('p').text.strip() if negative_comment else ""

            if positive_text:
                comments.append(positive_text)
                ratings.append('Positif')
                commenters.append(commenter)
                dates.append(date)

            if negative_text:
                comments.append(negative_text)
                ratings.append('Négatif')
                commenters.append(commenter)
                dates.append(date)

        except Exception as e:
            print(f"Error extracting a review: {e}")

    df = pd.DataFrame({
        'Commenter': commenters,
        'Date': dates,
        'Comments': comments,
        'Ratings': ratings
    })

    return df

# URLs to scrape
urls_to_scrape_train = [
    "https://www.hospitalidee.fr/etablissement/hopital-ambroise-pare-paris-ap-hp-hopitaux-idf-ile-de-france-ouest",
    "https://www.hospitalidee.fr/etablissement/hopital-cochin-paris-ap-hp-hopitaux-paris-centre",
    "https://www.hospitalidee.fr/etablissement/chu-lille",
    "https://www.hospitalidee.fr/etablissement/chu-lille-hopital",
    "https://www.hospitalidee.fr/etablissement/hopital-valenciennes",
    "https://www.hospitalidee.fr/etablissement/hopital-pitie-salpetriere-paris-ap-hp",
    "https://www.hospitalidee.fr/etablissement/hopital-de-frejus-saint-raphael",
    "https://www.hospitalidee.fr/etablissement/hopital-jean-marcel-de-brignoles",
    "https://www.hospitalidee.fr/etablissement/polyclinique-saint-jean",
    "https://www.hospitalidee.fr/etablissement/hopital-de-cavaillon-lauris",
    "https://www.hospitalidee.fr/etablissement/clinique-du-parc-lyon",
    "https://www.hospitalidee.fr/etablissement/chu-saint-etienne-hopital-nord",
    "https://www.hospitalidee.fr/etablissement/clinique-aemilie-de-vialar",
    "https://www.hospitalidee.fr/etablissement/chu-bordeaux-hopitaux-de-pellegrin",
    "https://www.hospitalidee.fr/etablissement/hopital-charles-perrens-bordeaux",
    "https://www.hospitalidee.fr/etablissement/ch-d-excideuil",
    "https://www.hospitalidee.fr/etablissement/hopital-perpignan-saint-jean",
    "https://www.hospitalidee.fr/etablissement/ch-narbonne",
    "https://www.hospitalidee.fr/etablissement/chu-caen",
    "https://www.hospitalidee.fr/etablissement/hopital-la-musse",
    "https://www.hospitalidee.fr/etablissement/hopital-jacques-monod-le-havre",
    "https://www.hospitalidee.fr/etablissement/chu-rouen-hopital-charles-nicolle",
    "https://www.hospitalidee.fr/etablissement/hopital-la-musse",
    "https://www.hospitalidee.fr/etablissement/hopital-henri-mondor-ap-hp",
    "https://www.hospitalidee.fr/etablissement/clinique-de-meudon-la-foret-3-5",
    "https://www.hospitalidee.fr/etablissement/chu-amiens-salouael",
    "https://www.hospitalidee.fr/etablissement/hopital-compiegne-noyon?avis"
]


urls_to_scrape_test = [
    "https://www.hospitalidee.fr/etablissement/hopital-creil",
    "https://www.hospitalidee.fr/etablissement/chu-nantes-hotel-dieu",
    "https://www.hospitalidee.fr/etablissement/hopital-prive-saint-martin"
]

############################### Scrapping data train #########################
# DataFrame to store all comments
combined_df = pd.DataFrame()

# Scraping each page and concatenating the results
for url in urls_to_scrape_train:
    df = scrape_page(url)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Displaying the combined DataFrame
print(combined_df)

############################### Scrapping data test #########################

df_test = pd.DataFrame()

# Scraping each page and concatenating the results
for url in urls_to_scrape_test:
    df2 = scrape_page(url)
    df_test = pd.concat([df_test, df2], ignore_index=True)


"""
########################### DATA MANAGEMENT ###################################
"""

# Téléchargement du modèle mBERT multilingue pour le français

module_url = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3"
bert_layer = hub.KerasLayer(module_url, trainable=True)

# Fonction de nettoyage de texte
def clean_text(text):
    text = text.lower()  # Mettre en minuscule
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if not unicodedata.combining(c))  # Retirer les accents
    translation_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # Retirer la ponctuation
    text = text.translate(translation_table)
    text = ''.join(c for c in text if not c.isdigit())  # Retirer les chiffres
    text = text.replace("’", " ")  # Retirer les apostrophes
    text = text.replace("rien", "")
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+', '', text)
    return text.strip()  # Retirer les espaces en début et fin de texte

# Chargement des données
data = combined_df
data_test = df_test

# Nettoyage des commentaires
data['clean'] = data['Comments'].apply(clean_text)
data = data[~data['clean'].str.strip().isin(['', '#', '# '])]

data_test['clean'] = data_test['Comments'].apply(clean_text)
data_test = data_test[~data_test['clean'].str.strip().isin(['', '#', '# '])]


#### Stopwords

def remove_stopwords_except(text):
    stop_words = set(stopwords.words('french'))
    additionnal_word = ['patient','patients','dire','peu','peut','surtout', 'ete', 'elles','meme','h','faire','quand','apres','car','merci', 'tous', 'sans','a','tout', 'comme', 'plus', 'tres', 'avoir', 'ca', 'etre', 'cette', 'si', 'fait']
    stop_words.update (additionnal_word)
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return filtered_words

data['tokens_commentaires'] = data['clean'].apply(remove_stopwords_except)

###############################################################################
######################### Initialisation du tokenizer BERT #####################
################################################################################

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def bert_encode(texts, tokenizer, max_len=512):
    all_input_ids = []
    all_attention_masks = []
    all_segment_ids = []

    for text in texts:
        # Tokenisation du texte
        tokens = tokenizer.tokenize(text)

        # Ajout des tokens [CLS] et [SEP]
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        # Limitation de la longueur maximale
        if len(tokens) > max_len:
            tokens = tokens[:max_len - 1] + ['[SEP]']

        # Convertir les tokens en IDs
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Calculer les masques d'attention
        attention_mask = [1] * len(input_ids)

        # Padding
        padding_length = max_len - len(input_ids)
        input_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        segment_ids = [0] * max_len  # BERT multilingue n'utilise pas segment_ids, mais il faut les fournir

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_segment_ids.append(segment_ids)

    return {
        'input_word_ids': np.array(all_input_ids),
        'input_mask': np.array(all_attention_masks),
        'input_type_ids': np.array(all_segment_ids)
    }

###################### Préparation des données d'entraînement et de test ##############
train = data
test = data_test


train_input = bert_encode(train['clean'].values, tokenizer, max_len=100)
test_input = bert_encode(test['clean'].values, tokenizer, max_len=100)

mapping = {'Positif': 1, 'Négatif': 0}

train['Ratings'] = train['Ratings'].map(mapping)
test['Ratings'] = test['Ratings'].map(mapping)



# Vérifiez s'il y a des valeurs manquantes ou infinies après le mapping
print(train[['Ratings']].isna().sum())
print(np.isinf(train[['Ratings']]).sum())

"""
# Supprimez les valeurs manquantes ou infinies
train = train.dropna(subset=['Ratings'])
train = train[~np.isinf(train['Ratings'])]
test = test.dropna(subset=['Ratings'])
test = test[~np.isinf(test['Ratings'])]
"""

#################################################
train_labels = train['Ratings'].values
test_labels = test['Ratings'].values


#######################################################################
############################# Model BERT ##############################
#######################################################################


def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_type_ids")

    bert_inputs = {'input_word_ids': input_word_ids, 'input_mask': input_mask, 'input_type_ids': segment_ids}
    pooled_output = bert_layer(bert_inputs)['pooled_output']
    clf_output = Dense(256, activation='relu')(pooled_output)
    clf_output = Dropout(0.3)(clf_output)
    out = Dense(1, activation='sigmoid')(clf_output)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(learning_rate=2e-5), loss='binary_crossentropy', metrics=['accuracy'])

    return model


############### lancement du model

model = build_model(bert_layer, max_len=100)
train_history = model.fit(
    [train_input['input_word_ids'], train_input['input_mask'], train_input['input_type_ids']],
    train_labels,
    validation_split=0.2,
    epochs=5,
    batch_size=32
)


#################### Visulisation des résultats du modèle ####################

predictions = model.predict(test_input)
rounded_predictions = np.round(predictions)
test['bert_predictions'] = rounded_predictions
test['bert_predictions'] = test['bert_predictions'].astype(int)
print(test[['clean', 'Ratings', 'Comments', 'bert_predictions']])



###############################################################################

######################## Mots clés / domaines à améliorer #####################

"""
Textblob
"""
#############################################################
#############################################################


from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer 

def calcul_sentiment(commentaire):    
    sentiment = TextBlob(commentaire,pos_tagger=PatternTagger(),analyzer=PatternAnalyzer()).sentiment[0]
    return sentiment

test['score_sentiment'] = test['Comments'].apply(calcul_sentiment)


"""
Résultats html
"""
resultat_final = test.to_html(escape=False)

with open('resultat_final.html', 'w', encoding='utf-8') as f:
    f.write(resultat_final)


"""
Nuage de mots en fonction de la fréquence des mots
"""

data_filtered = data[data['Ratings'] == 0] # spécifique aux commentaires négatifs 
mots_combines = data_filtered['tokens_commentaires'].explode()
mots_freq = mots_combines.value_counts()
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(mots_freq)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage de mots des critiques positives')
plt.show()