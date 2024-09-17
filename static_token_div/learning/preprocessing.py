""" This script is a pre-processing file for classifier
.module author:: Marius THORRE
"""
import numpy as np

from static_token_div.algorithms import w2v


def process(
    embedding_file_path: str,
    k: int
) -> np.ndarray:
    """
    Pre processing file
    :param embedding_file_path:
    :param k:
    :return:
    """

    embeddings = []

    with open(embedding_file_path, "r", encoding='utf-8') as file:
        for line in file:
            embeddings.append(line.split(' '))
    print(embeddings[:3])
    return np.zeros(0)
