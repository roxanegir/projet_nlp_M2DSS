# projet_nlp_M2DSS
 Analyse des sentiments des patients par Marie Tapia, Chafiaa Challal, Roxane Girault


# Analyse de Sentiments des Avis d'Hôpitaux en France

## Description

Ce projet vise à réaliser une analyse de sentiments sur des avis d'hôpitaux en France. Les avis sont classés en commentaires positifs et négatifs et sont analysés en utilisant le modèle mBERT et TextBlob. Un nuage de mots est également généré pour identifier les domaines à améliorer.

## Structure du Projet

1. **Scraping des Données**
    - Les avis sur les hôpitaux sont scrappés à partir du site Hospitalidee web en utilisant Selenium et BeautifulSoup.
    - Les avis sont classés en deux catégories : positifs et négatifs.
    - Les données scrappées sont stockées dans deux DataFrames : `combined_df` pour les données d'entraînement et `df_test` pour les données de test.

2. **Préparation des Données**
    - Les commentaires sont nettoyés avec la fonction 'clean_text' pour retirer les caractères spéciaux, les accents, les chiffres, et la ponctuation.
    - D'un autre côté pour les nuages de mots, un autres dataframe est crée duquel les stopwords sont retirés pour faciliter l'analyse.

3. **Entraînement du Modèle mBERT**
    - Le modèle mBERT est utilisé pour la classification des avis : en positifs(1) et négatifs(2).
    - Les avis sont tokenisés et préparés pour l'entraînement du modèle .
    - Le modèle est entraîné sur les données d'entraînement et testé sur les données de test.

4. **Analyse de Sentiment avec TextBlob**
    - Une analyse de sentiment supplémentaire est effectuée en utilisant TextBlob.
    - Les scores de sentiment varient de -1 (très négatif) à 1 (très positif).

5. **Identification des Domaines à Améliorer**
    - Un nuage de mots est généré à partir des commentaires négatifs pour identifier les domaines nécessitant des améliorations.
    - Les mots les plus fréquents dans les commentaires négatifs sont mis en évidence.

## Utilisation

Pour exécuter ce projet, suivez les étapes suivantes :

1. Clonez le dépôt du projet.
2. Installez les dépendances nécessaires listées dans `requirements.txt`.
3. Exécutez le script principal pour scraper les données, entraîner le modèle, et effectuer l'analyse de sentiments.

## Résultats

- Le modèle mBERT a été utilisé pour classer les avis en positifs et négatifs avec une précision acceptable.
- TextBlob a fourni une analyse de sentiment supplémentaire pour vérifier les résultats du modèle mBERT.
- Le nuage de mots a permis d'identifier les domaines où les hôpitaux peuvent améliorer leurs services en se basant sur les commentaires négatifs des patients.

## Conclusion

Ce projet démontre l'efficacité de l'utilisation de techniques de traitement du langage naturel (NLP) et de modèles de deep learning pour l'analyse de sentiments. Les résultats obtenus peuvent aider les hôpitaux à mieux comprendre les retours de leurs patients et à identifier les domaines nécessitant des améliorations.


