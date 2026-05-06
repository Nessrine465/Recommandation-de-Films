---

# Architecture du projet

```bash
Recommandation-de-Films/
│
├── data/
│   └── tmdb_5000_movies.csv
│
├── documents.json
├── index.faiss
├── context.txt
│
├── indexation.py
├── vectorisation.py
├── recherche.py
├── rag.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Fonctionnement du pipeline RAG

## Étape 1 — Préparation des données (`indexation.py`)

Le fichier CSV contenant les films est chargé puis transformé en documents textuels.

Chaque document contient :

- le titre
- la date de sortie
- le synopsis
- la note
- la langue originale
- les genres

Les documents sont ensuite sauvegardés dans :

```bash
documents.json
```

Lancer :

```bash
python3 indexation.py
```

---

## Étape 2 — Création des embeddings et de la base FAISS (`vectorisation.py`)

Les documents sont transformés en embeddings grâce au modèle :

```python
all-MiniLM-L6-v2
```

Les embeddings sont ensuite stockés dans une base vectorielle FAISS.

Lancer :

```bash
python3 vectorisation.py
```

Cela génère :

```bash
index.faiss
```

---

## Étape 3 — Recherche vectorielle (`recherche.py`)

La question utilisateur est transformée en embedding.

FAISS recherche ensuite les films les plus similaires.

Le script retourne :

- le score de similarité
- le titre du film
- les genres
- la note
- la date de sortie

Lancer :

```bash
python3 recherche.py
```

---

## Étape 4 — Système RAG complet (`rag.py`)

Le système final :

- récupère les chunks pertinents avec FAISS
- envoie le contexte au modèle Groq
- génère une réponse finale avec le LLM

Lancer :

```bash
python3 rag.py
```

---

# Fonctionnalités implémentées

## Fonctionnalités principales

- ingestion du CSV
- transformation des documents
- extraction des métadonnées JSON
- génération des embeddings
- base vectorielle FAISS
- recherche vectorielle
- intégration du LLM Groq
- interface interactive dans le terminal

---

# Bonus réalisés

## Bonus A — Historique de conversation

Le système conserve les derniers échanges de la conversation.

---

## Bonus B — Score de confiance

Le système affiche le score de similarité FAISS.

Si le score est trop mauvais, le système refuse de répondre :

```text
I did not find directly relevant information in my database.
```

---

## Bonus D — Comparaison de films

Le système peut comparer deux films.

Exemple :

```text
compare Avatar and Titanic
```

---

# Exemples de requêtes

```text
A movie about a hero saving the world
```

```text
compare Avatar and Titanic
```

---

# Installation du projet

## Cloner le projet

```bash
git clone <url_du_repository>

```

---

## Créer un environnement virtuel

### Mac / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Configurer la clé API Groq

Créer un fichier `.env` à la racine du projet :

```env
GROQ_API_KEY=votre_cle_api
```

---

# Limites du projet

- Le dataset contient principalement des films jusqu’en 2017
- Les recommandations dépendent de la qualité des embeddings
- Le système ne peut pas recommander des films absents de la base

---

# Dataset utilisé

TMDB 5000 Movie Dataset

Source :

https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

---
















































