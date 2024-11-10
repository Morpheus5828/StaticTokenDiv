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

t1 = os.path.join(project_path, "resources/tlnl_tp1_data/alexandre_dumas/La_Reine_Margot.tok")
t2 = os.path.join(project_path, "resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok")
t3 = os.path.join(project_path, "resources/tlnl_tp1_data/alexandre_dumas/Le_Vicomte_de_Bragelonne.tok")
t4 = os.path.join(project_path, "resources/tlnl_tp1_data/alexandre_dumas/Les_Trois_Mousquetaires.tok")
t5 = os.path.join(project_path, "resources/tlnl_tp1_data/alexandre_dumas/Vingt_ans_apres.tok")

all_text = [t1, t2, t3, t4, t5]

triplet_file = os.path.join(project_path, "resources/evaluation.txt")
embedding_path = os.path.join(project_path, "demos/embedding.txt")

start = time.time()

k = 3
L = 2
minc = 5
word_except = ["<s>", "</s>"]
learning_rate = 0.01
nb_iterations = 10
early_stop = 2
embedding_dim = 100

all_embedding = ""
for idx, t in enumerate(all_text):
    print(f"\tCurrent file extraction: {t}")

    current_embedding = w2v.word2vec(
        text_path=t,
        embedding_path=os.path.join(embedding_path, f"t_{idx}"),
        k=k,
        L=L,
        minc=minc,
        word_except=word_except,
        learning_rate=learning_rate,
        embedding_dim=embedding_dim,
        nb_iterations=nb_iterations,
        optmim_random_choice=True,
        early_stop=early_stop,
        save_embedding_file=False
    )
    all_embedding += current_embedding

    score = eval_w2v.evaluation(
        text_path=t,
        embedding_file=embedding_path,
        triplet_file=triplet_file,
        minc=minc,
        word_except=word_except
    )

with open(embedding_path, 'w', encoding="utf-8") as f:
    f.write(all_embedding)

end = time.time()

print(f"\tProcess time: {end - start:.2f} s")



