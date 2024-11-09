""" This script is a demo file of Word2Vec process
.module author:: Marius THORRE
"""

import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

import static_token_div.algorithms.w2v as w2v
import static_token_div.eval.eval_w2v as eval_w2v
import time
import matplotlib.pyplot as plt
import numpy as np

text_path = os.path.join(project_path, "resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok")
triplet_file = os.path.join(project_path, "resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.100.sim.txt")
embedding_path = os.path.join(project_path, "embedding.txt")

start = time.time()

k = 3
L = 2
minc = 5
word_except = ["<s>", "</s>"]
learning_rate = 0.01
nb_iterations = 5
early_stop = 2
embedding_dim = 100


def run(condition: bool, nb_iteration: int):
    acc = []
    for _ in range(nb_iteration):
        w2v.word2vec(
            text_path=text_path,
            embedding_path=embedding_path,
            k=k,
            L=L,
            minc=minc,
            word_except=word_except,
            learning_rate=learning_rate,
            embedding_dim=embedding_dim,
            nb_iterations=nb_iteration,
            optmim_random_choice=condition,
            early_stop=early_stop
        )

        a = eval_w2v.evaluation(
            text_path=text_path,
            embedding_file=embedding_path,
            triplet_file=triplet_file,
            minc=minc,
            word_except=word_except
        )

        acc.append(a)

    return acc


end = time.time()

print(f"\tProcess time: {end - start:.2f} s")

scores = run(condition=True, nb_iteration=10)
plt.plot(range(len(scores)), scores, c="blue", label=f"with adding noise, mean: {np.mean(scores):.2f}")
scores = run(condition=False, nb_iteration=10)
plt.plot(range(len(scores)), scores, c="red", label=f"without adding noise, mean: {np.mean(scores):.2f}")
plt.xlabel("Iterations")
plt.ylim(0, 100)
plt.ylabel("Scores")
plt.title("Word2Vec scores with various noise")
plt.legend()
plt.show()

