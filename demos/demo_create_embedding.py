""" This script is a demo file to generate embedding file
.module author:: Marius THORRE
"""

import static_token_div.algorithms.w2v as w2v
import time

text_path = "../resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok"
embedding_path = "../test/context_generated.txt"

L = 2  # number of positive and negative word context between target word
k = 2  # number of neg context for 1 pos context
eta = 0  # learning rate
e = 0  # nb iteration
minc = 5  # number of minimal take occurrence to take as target word
word_except = ["<s>", "</s>"]  # exception word, not take as target word

start = time.time()
print("\tStarting embedding generator file creation ...")
print("\tPlease hold on, it can take few moments :-)")

w2v.embedding_generator(
    save_path=embedding_path,
    text_path=text_path,
    L=L,
    k=k,
    eta=eta,
    e=e,
    minc=minc,
    word_except=word_except
)

end = time.time()
print(f"\tCreation process time: {end - start:.2f} s")
print(f"\tFile save at: {embedding_path}")
