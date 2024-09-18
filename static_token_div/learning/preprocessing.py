""" This script is a pre-processing file for classifier
.module author:: Marius THORRE
"""
import random
from static_token_div.algorithms import w2v
import numpy as np

def process(
    embedding_file_path: str,
    k: int
) -> None:
    """
    Pre processing file
    :param embedding_file_path:
    :param k:
    :return:
    """

    embeddings = []

    with open(embedding_file_path, "r", encoding='utf-8') as file:
        for line in file:
            line = line.strip().split(' ')
            embeddings.append(line)

    y = np.ones(len(embeddings))

    positive_context = embeddings
    negative_context = []
    negative_context_label = []
    # create negative context:
    for line in embeddings:
        target = line[len(line) // 2]
        i = 0
        while i < k:
            random_value = random.randint(0, len(embeddings) -1)

            if embeddings[random_value][len(line) // 2] != target:
                negative_embedding = embeddings[random_value][len(line) // 2] = target
                negative_context.append(negative_embedding)
                negative_context_label.append(-1)
                i +=1

    print(len(positive_context), len(negative_context))
