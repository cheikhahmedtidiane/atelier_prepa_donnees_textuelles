# Atelier : Préparation de Données Textuelles pour le Machine Learning & Deep Learning

L'objectif de cet atelier est la construction d'un pipeline complet et robuste de préparation et de traitement de données textuelles (NLP) avant l'entraînement de modèles prédictifs. 

Le cas d'étude repose sur la classification automatique de sentiments à partir d'un corpus d'avis clients réels mais très bruités (valeurs manquantes, doublons, URLs, casse hétérogène, emojis, et ponctuations répétées).

## Structure du Projet

Le projet respecte scrupuleusement l'architecture industrielle suivante :
```text
atelier_prepa_donnees_textuelles/
├── notebooks/
│   └── atelier_prepa_donnees_textuelles.ipynb   # Pipeline complet pas à pas
└── data/
    ├── smart_reviews_raw.csv                    # Dataset brut d'origine
    ├── smart_reviews_cleaned.csv                # Dataset nettoyé et normalisé
    └── tfidf_matrix.npz                         # Matrice de vectorisation compressée
```

## Étapes clés du Pipeline NLP

Le pipeline de traitement de texte est découpé en 6 grandes parties :

1. **Exploration du Corpus (EDA)** : Analyse des dimensions du dataset, identification des anomalies graphiques (URLs, mentions, hashtags), mesure de la longueur des textes et mise en évidence du fort déséquilibre de la variable cible (`sentiment`).
2. **Nettoyage Automatisé** : Suppression des lignes vides, élimination des doublons de texte, effacement des URLs et des mentions, et extraction sémantique des hashtags.
3. **Tokenisation Linguistique** : Découpage chirurgical du texte à l'aide de la bibliothèque `NLTK` (`word_tokenize` et `punkt_tab`) pour isoler les véritables unités de sens.
4. **Normalisation Avancée** : Uniformisation de la casse en minuscules, exclusion ciblée des *stop words* neutres tout en préservant les mots critiques (négations et adverbes d'intensité), et application d'un *French SnowballStemmer*.
5. **Découpage Stratifié** : Séparation hermétique des données en ensembles d'entraînement (80%) et de test (20%) avec maintien strict des proportions de classes pour éviter toute fuite de données (*Data Leakage*).
6. **Vectorisation Numérique** : Transformation du texte normalisé en vecteurs de caractéristiques. Comparaison entre les approches *Bag of Words*, *TF-IDF Classique* et *TF-IDF avec Unigrams & Bigrams*.

## Fonctionnalité Bonus d'Expert : Meta-Features Structurelles

Pour maximiser les performances du futur modèle de classification, le pipeline intègre une étape d'**Ingénierie de Caractéristiques Hybride** exécutée sur le texte brut. Quatre indicateurs structurels et comportementaux sont extraits en parallèle des mots :
* **`capitals_ratio`** : Proportion de lettres en majuscules pour capter l'intensité ou l'agressivité de l'avis.
* **`expressive_punct_density`** : Densité de points d'exclamation et d'interrogation (`!` et `?`), signaux forts d'excitation ou de mécontentement.
* **`lexical_diversity`** : Rapport de mots uniques pour identifier les copier-coller ou les textes redondants.
* **`has_negation`** : Indicateur binaire détectant la présence de structures clés de négation (*pas, plus, jamais, rien*), guidant le modèle sur les retournements de sens.

Ces caractéristiques complémentaires sont fusionnées horizontalement avec la matrice de termes à l'aide de `scipy.sparse.hstack` avant la modélisation.

## Dépendances requises

Pour exécuter le notebook, les packages Python suivants sont nécessaires :
```bash
pip install pandas numpy matplotlib seaborn nltk scipy scikit-learn
```