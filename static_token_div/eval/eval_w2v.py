""" This script is an evaluation file
.module author:: Marius THORRE
"""

import static_token_div.tools.tools as tools
import numpy as np


def evaluation(
        div_file_path: str,
        eval_file_path: str
) -> float:
    """
    This file evaluate
    :param div_file_path: div file generated path using w2v.py file
    :param eval_file_path: evaluation file path in txt
    :return:
    """
    correct_count = 0
    total_count = 0

    with open(eval_file_path, 'r') as f:
        embeddings = load_embeddings(div_file_path)
        for line in f:
            m, m_pos, m_neg = line.split()

            vec_m = embeddings[m]
            vec_m_pos = embeddings[m_pos]
            vec_m_neg = embeddings[m_neg]

            sim_pos = tools.cosine_similarity(vec_m, vec_m_pos)
            sim_neg = tools.cosine_similarity(vec_m, vec_m_neg)

            if sim_pos > sim_neg:
                correct_count += 1
            total_count += 1

    return correct_count / total_count


def load_embeddings(embedding_file):
    embeddings = {}
    with open(embedding_file, 'r') as f:
        for line in f:
            elements = line.split()
            word = elements[0]
            vector = np.array([float(x) for x in elements[1:]])
            embeddings[word] = vector
    return embeddings
