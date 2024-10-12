""" This script is a demo file to generate embedding file
.module author:: Marius THORRE
"""

import static_token_div.algorithms.w2v as w2v
import static_token_div.eval.eval_w2v as eval_w2v
import time

text_path = "../resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok"
triplet_file = "../resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.100.sim.txt"
embedding_path = "embedding.txt"
word_except = ["<s>", "</s>"]

start = time.time()


w2v.word2vec(
    text_path=text_path,
    embedding_path=embedding_path,
    k=10,
    L=2,
    minc=5,
    word_except=word_except,
    learning_rate=0.1,
    embedding_dim=100,
    nb_iterations=5
)

eval_w2v.evaluation(
    text_path=text_path,
    embedding_file=embedding_path,
    triplet_file=triplet_file,
    minc=5,
    word_except=word_except
)


end = time.time()

print(f"\tProcess time: {end - start:.2f} s")

