""" This script is a demo file of Word2Vec process but with different parameter
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
import matplotlib.pyplot as plt

text_path = os.path.join(project_path, "resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok")
triplet_file = os.path.join(project_path, "resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.100.sim.txt")
embedding_path = os.path.join(project_path, "emb_filename.txt")


L = 2
minc = 5
word_except = ["<s>", "</s>"]
learning_rate = 0.01
nb_iterations = 5
early_stop = 2
embedding_dim = 100


def run(noise_condition: bool, nb_iteration: int = 5, k_max: int = 10, k_min: int = 1):
    scores = []
    for k in range(k_min, k_max):
        current_score = 0
        for i in range(nb_iteration):
            print(f"\n k: {k}, i: {i}")
            w2v.word2vec(
                text_path=text_path,
                embedding_path=embedding_path,
                k=k,
                L=L,
                minc=minc,
                word_except=word_except,
                learning_rate=learning_rate,
                embedding_dim=embedding_dim,
                nb_iterations=nb_iterations,
                optmim_random_choice=noise_condition,
                early_stop=early_stop
            )

            score = eval_w2v.evaluation(
                text_path=text_path,
                embedding_file=embedding_path,
                triplet_file=triplet_file,
                minc=minc,
                word_except=word_except
            )
            current_score += score

        scores.append(current_score/nb_iteration)

        print(current_score/nb_iteration)

    return scores


times = run(noise_condition=True)
plt.plot(range(len(times)), times, c="blue", label="with adding noise")
scores = run(noise_condition=False)
plt.plot(range(len(scores)), scores, c="red", label="without adding noise")
plt.xlabel("k")
plt.ylabel("Scores")
plt.title("Word2Vec scores with various k")
plt.legend()
plt.show()




