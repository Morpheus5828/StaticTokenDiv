import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Embedding:
    def __init__(self, file_path):
        self.words = {}
        self.file_path = file_path
        self._extract_file()

    def _extract_file(self):
        with open(self.file_path, "r", encoding="utf8", errors="ignore") as f:
            for idx, s in enumerate(f.readlines()):
                if idx > 0:
                    line = s.replace("\n", "").split(" ")
                    line.remove("")
                    self.words[line[0]] = np.array(line[1:], dtype=float)
        return self.words

    def knn(self, vector, k=5):
        distances = []

        for word, embed in self.words.items():
            distance = np.linalg.norm(embed - vector)
            distances.append((word, distance))

        distances.sort(key=lambda x: x[1])
        closest_words = [word for word, _ in distances[:k]]

        return closest_words

    def acp(self, word_list):
        X = np.array([self.words[v] for v in word_list])
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
        result = {word_list[i]: X[i] for i in range(len(word_list))}
        return result

    def plot_acp(self, data):
        x, y, name = [], [], []
        for point in data.keys():
            name.append(point)
            x.append(data[point][0])
            y.append(data[point][1])

        fig, ax = plt.subplots()
        ax.scatter(x, y)

        for i, txt in enumerate(name):
            ax.annotate(
                txt,
                (x[i], y[i]),
                xytext=(5, 5),
                textcoords="offset points"
            )

        plt.show()







