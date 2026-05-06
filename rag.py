import os

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from recherche import charger_documents, charger_index, rechercher

# Charge les variables d'environnement depuis le fichier .env
load_dotenv()

# Initialise le client Groq avec la clé API
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def read_file(file_path):

    # Lit le contenu d'un fichier texte 
    # Utilisé ici pour charger le template du prompt système depuis contexte.txt
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
    
def construire_prompt_systeme(chunks_pertinents):

    # Construit le prompt système envoyé au Llm
    #  Le fichier context.txt contient les règles générales du RAG

    context = read_file("context.txt")

    chunks_text = ""
    
    # Transforme les chunks récupérés en une section de contexte lisible pour le LLM
    for i, chunk in enumerate(chunks_pertinents, start=1):
        chunks_text += f"""
Movie {i}:
Title: {chunk["metadata"]["title"]}
Genres: {chunk["metadata"]["genres"]}
Rating: {chunk["metadata"]["rating"]}
Release date: {chunk["metadata"]["release_date"]}
Content:
{chunk["content"]}
"""
    # Remplace la varibale {{chunks}} dans context.txt par les vrais chunks 
    full_context = context.replace("{{chunks}}", chunks_text)

    return full_context

def generer_reponse(question, chunks_pertinents, historique):
    #  Génère une réponse avec Groq à partir de la question utilisateur, des chunks pertinents récupérés par Faiss et de l'historique de la conversation

    historique_text = ""

    # On garde seulement les 3 derniers échanges de la conversation pour ne pas dépasser les limites de tokens du LLM
    for item in historique[-3:]:
        historique_text += f"""
Previous user question:
{item["question"]}

Previous assistant answer:
{item["answer"]}
"""

    prompt = f"""
Conversation history:
{historique_text}

Current user question:
{question}

Answer the user based only on the context provided in the system message.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": construire_prompt_systeme(chunks_pertinents)},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

def trouver_film_par_titre(titre, documents):

    # Recherche un film exact dans la base de connaissance à partir de son titre
    titre = titre.lower()

    for document in documents:
        movie_title = document["metadata"]["title"].lower()

        if titre == movie_title:
            return document

    return None

def comparer_films(titre1, titre2, documents):

    # Compare deux films en utilisant les informations présentes dans la base de connaissance
    # Bonus D : mode comparaison

    film1 = trouver_film_par_titre(titre1, documents)
    film2 = trouver_film_par_titre(titre2, documents)

    if film1 is None or film2 is None:
        return "I could not find one of the movies in the database."

    prompt = f"""
Compare these two movies using only the information provided.

Movie 1:
Title: {film1["metadata"]["title"]}
Genres: {film1["metadata"]["genres"]}
Rating: {film1["metadata"]["rating"]}
Release date: {film1["metadata"]["release_date"]}
Content:
{film1["content"]}

Movie 2:
Title: {film2["metadata"]["title"]}
Genres: {film2["metadata"]["genres"]}
Rating: {film2["metadata"]["rating"]}
Release date: {film2["metadata"]["release_date"]}
Content:
{film2["content"]}

Provide a clear comparison:
- common points
- differences
- which one is better depending on the viewer's preference
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a movie comparison assistant. Do not invent information."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

def main():

    # Fonction principale : charge la base de vectorielle, lance une boucle interactive, récupère les chunks pertinents, génère une réponse avec Groq

    print("Chargement de la base de connaissance...")

    documents = charger_documents()
    index = charger_index()

    # Le modèle doit etre le même que celui utilisé dans vectorisation.py
    modele = SentenceTransformer("all-MiniLM-L6-v2")

    print("Système RAG prêt. Tapez 'quit' pour quitter.\n")
    
    historique = [] 

    # Seuil utilisé pour le bonus B: score de confiance
    #  Avec IndexFlatL2, plus le score est petit, plus le résultat est pertinent.

    SEUIL_CONFIANCE = 1.5
    
    while True:
        question = input("Enter your movie request : ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        # Bonus D : mode comparaison

        if question.lower().startswith("compare "):
            try:
                comparaison = question[8:]
                titre1, titre2 = comparaison.split(" and ")

                reponse = comparer_films(titre1.strip(), titre2.strip(), documents)

                print("\nComparison:\n")
                print(reponse)
                print("\n" + "-" * 50 + "\n")
                continue

            except ValueError:
                print("\nPlease use this format:")
                print("compare Avatar and Titanic")
                print("\n" + "-" * 50 + "\n")
                continue

        
        # Recherche des chunks pertinents avec Faiss
        resultats = rechercher(question, modele, index, documents, k=5)

        if len(resultats) == 0:
            print("\nNo relevant results found.")
            print("\n" + "-" * 50 + "\n")
            continue

        meilleur_score = resultats[0]["score"]

        print(f"\nBest FAISS score: {meilleur_score}")

        # Bonus B : si le meilleur score est trop mauvais, on refuse de répondre 
        if meilleur_score > SEUIL_CONFIANCE:
            print("\nWarning:")
            print("I did not find directly relevant information in my database.")
            print("\nSources used:")
            for i, resultat in enumerate(resultats, start=1):
                print(f"{i}. {resultat['metadata']['title']} - score: {resultat['score']}")

            print("\n" + "-" * 50 + "\n")
            continue

         # Génération de la réponse uniquement si les résultats trouvés sont suffisamment pertinents
        reponse = generer_reponse(question, resultats, historique)


        print("\nAnswer:\n")
        print(reponse)

         # Ajoute l'échange dans l'historique 
         
        historique.append({
            "question": question,
            "answer": reponse
        })

        print("\nSources used:")
        for i, resultat in enumerate(resultats, start=1):
            print(f"{i}. {resultat['metadata']['title']} - score : {resultat['score']}")

        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()