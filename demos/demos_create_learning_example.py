""" This script is a demo file to generate embedding file
.module author:: Marius THORRE
"""

import static_token_div.tools.w2v_tools as w2v_tools
import time

text_path = "../resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok"
saving_path = "../static_token_div/algorithms/learning_file.txt"
word_except = ["<s>", "</s>"]

start = time.time()
print("\tStarting embedding generator file creation ...")
print("\tPlease hold on, it can take few moments :-)")

'''w2v_tools.generate_embeddings_file(
    text_path=text_path,
    save_path=saving_path,
    word_except=word_except,
    k=3,
    L=2,
    minc=5
)'''

a = w2v_tools._create_learning_file(
    text_path=text_path,
    word_except=word_except,
    k=3,
    L=2,
    minc=5
)


end = time.time()
print(f"\tCreation process time: {end - start:.2f} s")
#print(f"\tFile save at: {saving_path}")
