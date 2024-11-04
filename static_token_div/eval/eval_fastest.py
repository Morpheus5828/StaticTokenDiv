""" This script is Fastest evaluation file
.module author:: Marius THORRE
"""

import numpy as np
import pandas as pd
from static_token_div.tools.tools import cosine_similarity, get_text
from static_token_div.tools.embedding_tools import get_embedding_from_trigram, create_n_gram
from static_token_div.tools.w2v_tools import create_vocabulary


def evaluation(
    text_path: str,
    triplet_file: str,
    embedding_file: str,
    minc: int,
    n: int,
    word_except: list
) -> float:
    """
    Compute score of Word2Vec
    :param text_path: text path
    :param triplet_file: validation test file path
    :param embedding_file: embedding file generated
    :param minc: minimal occurrence number
    :param word_except: word except, will not appear as embedding target
    :return: score computed
    """

    text = get_text(text_path)
    vocab, _, _ = create_vocabulary(
        text=text,
        minc=minc,
        word_except=word_except,
        add_n_gram=True,
        n=n
    )

    embeddings = {}

    with open(embedding_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            word_idx = int(parts[0])
            vector = np.array([float(x) for x in parts[1:]])
            embeddings[word_idx] = vector

    eval_data = pd.read_csv(triplet_file, sep=" ", header=None, names=["m", "m_plus", "m_minus"])
    success_count = 0

    for _, row in eval_data.iterrows():
        m, m_plus, m_minus = vocab[row["m"]][0], vocab[row["m_plus"]][0], vocab[row["m_minus"]][0]

        m = get_embedding_from_trigram(vocab, embeddings, m)
        #m_plus = get_embedding_from_trigram(vocab, embeddings, m_plus)
        #m_minus = get_embedding_from_trigram(vocab, embeddings, m_minus)

        sim_m_plus = cosine_similarity(m, embeddings[m_plus])
        sim_m_minus = cosine_similarity(m, embeddings[m_minus])
        if sim_m_plus > sim_m_minus:
            success_count += 1

    accuracy = (success_count / len(eval_data)) * 100
    print(f"\tScore : {accuracy:.2f}%")
    return accuracy


