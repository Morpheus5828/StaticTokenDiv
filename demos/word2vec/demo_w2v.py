""" This script is a demo file of Word2Vec process
.module author:: Marius THORRE
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from static_token_div.eval import eval_w2v
import static_token_div.algorithms.w2v as w2v
import time

text_path = os.path.join(project_path, "resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.txt")
triplet_file = os.path.join(project_path, "resources/evaluation.txt")
embedding_path = os.path.join(project_path, "demos/Le_comte_de_Monte_Cristo_emb_filename.txt")

start = time.time()

k = 3
L = 2
minc = 5
word_except = ["<s>", "</s>"]
learning_rate = 0.01
nb_iterations = 5
early_stop = 2
embedding_dim = 100

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
    optmim_random_choice=True,
    early_stop=early_stop
)

score = eval_w2v.evaluation(
    text_path=text_path,
    embedding_file=embedding_path,
    triplet_file=triplet_file,
    minc=minc,
    word_except=word_except
)

print(f"\tEmbedding file save here: {embedding_path}")

end = time.time()

print(f"\tProcess time: {end - start:.2f} s")



