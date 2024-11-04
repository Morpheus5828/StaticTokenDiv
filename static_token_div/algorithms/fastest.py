""" This script is an implementation of Fastest algorithm from this paper
Enriching Word Vectors with Subword Information by Tomas Mikolov and al. June 2017
.module author:: Marius THORRE
"""

import numpy as np
import random
from static_token_div.tools.tools import sigmoid
from static_token_div.tools.fastest_tools import create_learning_file_with_n_gram


def fastest(
        text_path: str,
        embedding_path: str,
        k: int,
        L: int,
        n: int,
        minc: int,
        word_except: list,
        learning_rate: float,
        embedding_dim: int,
        nb_iterations: int = 10,
        optmim_random_choice: bool = False,
        early_stop: int = 1
) -> None:
    print("\tStart data extraction ...")
    print("\tPlease hold on, it can take few moments :-)")

    # 1 Extract data
    data = create_learning_file_with_n_gram(
        text_path=text_path,
        word_except=word_except,
        k=k,
        L=L,
        n=n,
        minc=minc,
        optmim_random_choice=optmim_random_choice
    )

    data = np.array(data)

    print(f"\ttarget size {1}, c_pos size: {1} c_neg size: {data.shape[1] - 2}")

    # 2 Run process
    W = np.random.rand(np.max(data) + 1, embedding_dim)
    C = np.random.rand(np.max(data) + 1, embedding_dim)
    print(f"\tStart process ... ")

    losses = []

    for iteration in range(nb_iterations):
        total_loss = 0
        random.shuffle(data)

        for data_point in data:
            target_idx = data_point[0]
            context_idx = data_point[1]
            negative_indices = data_point[2:]

            m = W[target_idx]
            c_pos = C[context_idx]

            score_pos = np.dot(m, c_pos)
            pred_pos = sigmoid(score_pos)
            loss_pos = -np.log(pred_pos + 1e-7)
            total_loss += loss_pos

            grad_m_pos = (pred_pos - 1) * c_pos
            grad_c_pos = (pred_pos - 1) * m

            W[target_idx] -= learning_rate * grad_m_pos
            C[context_idx] -= learning_rate * grad_c_pos

            for neg_idx in negative_indices:
                c_neg = C[neg_idx]
                score_neg = np.dot(m, c_neg)
                pred_neg = sigmoid(score_neg)
                loss_neg = -np.log(1 - pred_neg + 1e-7)
                total_loss += loss_neg
                grad_m_neg = pred_neg * c_neg
                grad_c_neg = pred_neg * m

                W[target_idx] -= learning_rate * grad_m_neg
                C[neg_idx] -= learning_rate * grad_c_neg

        avg_loss = total_loss / len(data)
        losses.append(avg_loss)
        print(f"\tIteration: {iteration + 1}/{nb_iterations}, Loss: {avg_loss:.6f}")

        if iteration >= early_stop:
            if losses[-1] > losses[-early_stop]:
                print(f"\tEarly stopping: loss increased from {losses[-early_stop]:.6f} to {losses[-1]:.6f}")
                break

    print("\tSaving embeddings...")
    with open(embedding_path, 'w', encoding='utf-8') as f:
        f.write(f"{W.shape[0]} {embedding_dim}\n")
        for idx, embedding in enumerate(W):
            embedding_str = ' '.join(map(str, embedding))
            f.write(f"{idx} {embedding_str}\n")

    print(f"\tEmbeddings saved to '{embedding_path}'")
