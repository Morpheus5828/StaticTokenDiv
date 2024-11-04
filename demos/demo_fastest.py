""" This script is a demo file of Fastest process
.module author:: Marius THORRE
"""

import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../'))
if project_path not in sys.path:
    sys.path.append(project_path)

import static_token_div.algorithms.fastest as fastest
import static_token_div.eval.eval_fastest as eval_fastest

text_path = os.path.join(project_path, "resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok")
triplet_file = os.path.join(project_path, "resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.100.sim.txt")
embedding_path = os.path.join(project_path, "embedding_fastest.txt")

L = 2
k = 5
minc = 5
n_gram_size = 3
word_except = ["<s>", "</s>"]
learning_rate = 0.01
nb_iterations = 30
early_stop = 2
embedding_dim = 100

fastest.fastest(
    text_path=text_path,
    embedding_path=embedding_path,
    k=k,
    L=L,
    n=n_gram_size,
    minc=minc,
    word_except=word_except,
    learning_rate=learning_rate,
    embedding_dim=embedding_dim,
    nb_iterations=nb_iterations,
    optmim_random_choice=True,
    early_stop=early_stop
)

a = eval_fastest.evaluation(
    text_path=text_path,
    embedding_file=embedding_path,
    triplet_file=triplet_file,
    minc=minc,
    n=n_gram_size,
    word_except=word_except
)