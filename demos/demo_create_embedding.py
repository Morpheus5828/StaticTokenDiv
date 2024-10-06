""" This script is a demo file to generate embedding file
.module author:: Marius THORRE
"""

import static_token_div.algorithms.w2v as w2v
import time

text_path = "../resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.train.tok"
vocab_path = "../static_token_div/learning/vocab.txt"
learning_path = "../static_token_div/learning/learning_file.txt"

L = 2  # number of positive and negative word context between target word
k = 2  # number of neg context for 1 pos context
minc = 5  # number of minimal take occurrence to take as target word
word_except = ["<s>", "</s>"]  # exception word, not take as target word
max_context = 100

start = time.time()
print("\tStarting embedding generator file creation ...")
print("\tPlease hold on, it can take few moments :-)")

w2v.embedding_generator(
    vocab_path=vocab_path,
    embeddings_path=learning_path,
    text_path=text_path,
    L=L,
    k=k,
    minc=minc,
    word_except=word_except,
    max_context=max_context
)

end = time.time()
print(f"\tCreation process time: {end - start:.2f} s")
print(f"\tFile save at: {learning_path}")
