""" This script is Word2Vec evaluation file
.module author:: Marius THORRE
"""

import numpy as np
import pandas as pd
from static_token_div.tools.tools import cosine_similarity, get_text
from static_token_div.tools.w2v_tools import create_vocabulary


def evaluation(
    text_path: str,
    triplet_file: str,
    embedding_file: str,
    minc: int,
    word_except: list,
    add_n_gram: bool = False,
    n: int = 3
) -> float:
    """
    Compute score of Word2Vec based on triplet validation.

    :param text_path: Path to the training text.
    :param triplet_file: Path to the triplet validation file.
    :param embedding_file: Path to the generated embedding file.
    :param minc: Minimal occurrence number.
    :param word_except: Words to exclude from the vocabulary.
    :param add_n_gram: Whether to include n-grams in the vocabulary.
    :param n: Size of n-grams.
    :return: Accuracy score in percentage.
    """
    # 1. Créer le vocabulaire
    text = get_text(text_path)
    vocab, word_counts_in_vocab, total_word_count = create_vocabulary(
        text=text,
        minc=minc,
        word_except=word_except,
        add_n_gram=add_n_gram,
        n=n
    )

    # 2. Lire les embeddings
    embeddings = {}
    with open(embedding_file, "r", encoding='utf-8') as f:
        header = next(f)  # Lire et ignorer la première ligne contenant les dimensions
        for line_num, line in enumerate(f, start=2):
            parts = line.strip().split()
            if len(parts) < 2:
                print(f"Warning: Ligne {line_num} mal formée dans le fichier d'embeddings. Ignorée.")
                continue  # Ignorer les lignes mal formées
            word = parts[0]
            try:
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float64)
                embeddings[word] = vector
            except ValueError:
                print(f"Warning: Erreur de conversion des valeurs numériques pour le mot '{word}' à la ligne {line_num}. Ignorée.")
                continue  # Ignorer les lignes avec des erreurs de conversion

    # Vérification des embeddings manquants
    missing_embeddings = [word for word in vocab if word not in embeddings]
    if missing_embeddings:
        print(f"Warning: {len(missing_embeddings)} mots manquent dans les embeddings.")

    # 3. Lire les triplets de validation
    eval_data = pd.read_csv(triplet_file, sep=" ", header=None, names=["m", "m_plus", "m_minus"])

    success_count = 0
    total = 0

    for index, row in eval_data.iterrows():
        m = row["m"]
        m_plus = row["m_plus"]
        m_minus = row["m_minus"]

        # Vérifier la présence des mots dans le vocabulaire et les embeddings
        if m not in vocab or m_plus not in vocab or m_minus not in vocab:
            print(f"Ignored triplet ({m}, {m_plus}, {m_minus}) car un ou plusieurs mots ne sont pas dans le vocabulaire.")
            continue  # Ignorer ce triplet si un des mots n'est pas dans le vocabulaire

        if m not in embeddings or m_plus not in embeddings or m_minus not in embeddings:
            print(f"Ignored triplet ({m}, {m_plus}, {m_minus}) car un ou plusieurs embeddings sont manquants.")
            continue  # Ignorer ce triplet si un des mots n'a pas d'embedding

        vec_m = embeddings[m]
        vec_m_plus = embeddings[m_plus]
        vec_m_minus = embeddings[m_minus]

        # Calcul des similarités cosinus
        sim_m_plus = cosine_similarity(vec_m, vec_m_plus)
        sim_m_minus = cosine_similarity(vec_m, vec_m_minus)

        if sim_m_plus > sim_m_minus:
            success_count += 1

        total += 1

    if total == 0:
        print("\tAucun triplet valide trouvé pour l'évaluation.")
        return 0.0

    accuracy = (success_count / total) * 100
    print(f"\tScore : {accuracy:.2f}%")
    return accuracy
