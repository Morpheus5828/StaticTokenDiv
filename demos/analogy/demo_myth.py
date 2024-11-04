""" This script is a demo file to watch Embedding myth in section 3.2 of the repport.
.module author:: Marius THORRE
"""

import os
import sys
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from static_token_div.eval.eval_analogy import Embedding

eval_path = text_path = os.path.join(project_path, "resources/model.txt")

print(f"\n\tExtract embedding from eval file: {eval_path}")
print("\tPlease hold it'can take few moment :-)")

embedding = Embedding(eval_path)

roi = embedding.words["roi"]
homme = embedding.words["homme"]
femme = embedding.words["femme"]

sum = roi - homme + femme
similars = embedding.knn(sum)

print(f"\tWords similar: {similars} to operation: roi - homme + femme")

dico = embedding.acp(similars)
d = {"roi": roi[:2], "homme": homme[:2], "femme": femme[:2]}
d.update(dico)

plt.scatter(d["reine"][0], d["reine"][1])
plt.text(d["reine"][0] + 0.07, d["reine"][1] + 0.0002, "Reine")

plt.scatter(d["waccho"][0], d["waccho"][1])
plt.text(d["waccho"][0] + 0.009, d["waccho"][1] + 0.1, "Waccho")

plt.scatter(d["goswinthe"][0], d["goswinthe"][1])
plt.text(d["goswinthe"][0] - 0.1, d["goswinthe"][1] - 0.2, "Goswinthe")

plt.scatter(d["teutberge"][0], d["teutberge"][1])
plt.text(d["teutberge"][0] - 0.25, d["teutberge"][1] + 0.17, "Teutberge")


plt.scatter(d["roi"][0], d["roi"][1])
plt.text(d["roi"][0] + 0.07, d["roi"][1] + 0.0002, "Roi")
plt.scatter(d["femme"][0], d["femme"][1])
plt.text(d["femme"][0] + 0.06, d["femme"][1] + 0.0002, "Femme")
plt.scatter(d["homme"][0], d["homme"][1])
plt.text(d["homme"][0] + 0.06, d["homme"][1], "Homme")

a = d["roi"] - d["homme"] + d["femme"]
plt.scatter(a[0], a[1])
plt.text(a[0] + 0.07, a[1] + 0.0002, "Sum")

plt.xlim(-1.2, 2)
plt.title("2D PCA on Word Embeddings")
plt.xlabel("First dimension")
plt.ylabel("Second dimension")
plt.ylim(-2, 2)

plt.show()

