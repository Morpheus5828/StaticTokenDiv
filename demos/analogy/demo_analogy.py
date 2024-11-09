""" This script is a demo file to watch Embedding myth in section 3.2 of the report.
But on many example, store in resource/analogies.txt.txt file.
.module author:: Marius THORRE
"""

import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from static_token_div.eval.eval_analogy import Embedding
analogie_path = os.path.join(project_path, "resources/analogies.txt.txt")
eval_path = os.path.join(project_path, "resources/model.txt")


print(f"\n\tExtract analogies.txt from storage file: {analogie_path}")
print("\tPlease hold it'can take few moment :-)")

with open(analogie_path, "r", encoding="utf8") as f:
    src, target = [], []
    for line in f.readlines():
        line = line.replace("\t", ",").replace("\n", "").split(" ")
        src.append(line[:-1])
        target.append(line[-1])


print("\tStart comparison process ...")

embedding = Embedding(file_path=eval_path)
score = 0
for idx, line in enumerate(src):
    word_embedding = []
    for word in line:
        word_embedding.append(embedding.words[word.lower()])

    sum = word_embedding[0] - word_embedding[1] + word_embedding[2]
    if target[idx] in embedding.knn(sum):
        score +=1

print(f"Score: {score/len(src)}")