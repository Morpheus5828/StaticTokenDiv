""" This script is an implementation of Word2Vec algorithm
.module author:: Marius THORRE
"""

import numpy as np
from static_token_div.tools.tools import sigmoid
from static_token_div.tools.w2v_tools import create_learning_file
from static_token_div.tools.tools import loss_function


def word2vec(
    text_path: str,
    embedding_path: str,
    k: int,
    L: int,
    minc: int,
    word_except: list,
    learning_rate: float,
    embedding_dim: int,
    nb_iterations: int = 10,
) -> None:
    print("\tStart data extraction ...")
    print("\tPlease hold on, it can take few moments :-)")

    # 1. Extract data
    data = create_learning_file(
        text_path=text_path,
        word_except=word_except,
        k=k,
        L=L,
        minc=minc
    )

    data = np.array(data)

    print(f"\ttarget size {1}, c_pos size: {1} c_neg size: {data.shape[1] - 2}")

    # 2 Run process
    W = np.random.rand(np.unique(data).shape[0] + 1, embedding_dim)
    C = np.random.rand(np.unique(data).shape[0] + 1, embedding_dim)
    print(f"\tStart process ... ")
    for i in range(nb_iterations):
        losses = []

        for line in data:
            m_idx = line[0]
            cpos_idx = line[1]
            cneg_idx = line[2:]

            m = W[m_idx]
            cpos = C[cpos_idx]
            cneg = np.array([C[i] for i in cneg_idx])

            losses.append(loss_function(m, cpos, cneg))
            grad_pos = (sigmoid(np.dot(m, cpos)) - 1) * m
            C[cpos_idx] -= learning_rate * grad_pos

            for idx in cneg_idx:
                cneg_vec = C[idx]
                grad_neg = sigmoid(np.dot(m, cneg_vec)) * m
                C[idx] -= learning_rate * grad_neg

            grad_target_pos = (sigmoid(np.dot(m, cpos)) - 1) * cpos
            grad_target_neg = np.sum([sigmoid(np.dot(m, C[i])) * C[i] for i in cneg_idx], axis=0)
            W[m_idx] -= learning_rate * (grad_target_pos + grad_target_neg)

        print(f"\t Iteration: {i+1}/{nb_iterations}, loss: {losses[i]}")

    # 3 save W
    with open(embedding_path, 'w') as f:
        f.write(f"{W.shape[0]} {embedding_dim}\n")

        for idx, embedding in enumerate(W):
            embedding_str = ' '.join(map(str, embedding))
            f.write(f"{idx} {embedding_str}\n")

    print("\tEmbeddings saved to 'embedding.txt'")

