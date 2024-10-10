""" This script is a demo file to generate embedding file
.module author:: Marius THORRE
"""

from static_token_div.tools.embedding_tools import embedding_generator
import time

text_path = "../resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok"
learning_path = "../static_token_div/learning_file.txt"

L = 2  # number of positive and negative word context between target word
k = 10  # number of neg context for 1 pos context
minc = 5  # number of minimal take occurrence to take as target word
word_except = ["<s>", "</s>"]  # exception word, not take as target word

start = time.time()
print("\tStarting embedding generator file creation ...")
print("\tPlease hold on, it can take few moments :-)")

embedding_generator(
    saving_path=learning_path,
    text_path=text_path,
    L=L,
    k=k,
    minc=minc,
    word_except=word_except
)

end = time.time()
print(f"\tCreation process time: {end - start:.2f} s")
print(f"\tFile save at: {learning_path}")
