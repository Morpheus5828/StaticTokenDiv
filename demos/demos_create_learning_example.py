""" This script is a demo file to generate embedding file
.module author:: Marius THORRE
"""

import static_token_div.tools.tools as tools
import time

text_path = "../resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.train.tok"
saving_path = "../static_token_div/algorithms/learning_file.txt"
word_except = ["<s>", "</s>"]

start = time.time()
print("\tStarting embedding generator file creation ...")
print("\tPlease hold on, it can take few moments :-)")

tools.generate_embeddings_file(
    text=tools.get_text(text_path),
    save_path=saving_path,
    word_except=word_except,
    k=2,
    L=100
)

end = time.time()
print(f"\tCreation process time: {end - start:.2f} s")
print(f"\tFile save at: {saving_path}")
