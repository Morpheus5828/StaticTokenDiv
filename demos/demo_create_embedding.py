""" This script is a demo file to generate embedding file
.module author:: Marius THORRE
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from static_token_div.tools.embedding_tools import embedding_generator
import time

text_path = os.path.join(project_path, "resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok")
learning_path = os.path.join(project_path, "../static_token_div/learning_file.txt")

L = 2
k = 10
minc = 5
word_except = ["<s>", "</s>"]

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
