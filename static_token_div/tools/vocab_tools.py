import torch


import torch

class Vocab:
    def __init__(self, **kwargs):
        self.dico_voca = {"<unk>": 0}
        self.word_array = ["<unk>"]
        indice_mot = 1

        if "emb_filename" in kwargs:
            with open(kwargs["emb_filename"], 'r', encoding="utf-8") as fi:
                ligne = fi.readline().strip()
                (self.vocab_size, self.emb_dim) = map(int, ligne.split(" "))
                self.matrice = torch.zeros((self.vocab_size + 1, self.emb_dim))  # +1 pour <unk>

                ligne = fi.readline().strip()
                while ligne != '':
                    splitted_ligne = ligne.split()
                    self.dico_voca[splitted_ligne[0]] = indice_mot
                    self.word_array.append(splitted_ligne[0])

                    for i in range(1, len(splitted_ligne)):
                        self.matrice[indice_mot, i - 1] = float(splitted_ligne[i])
                    indice_mot += 1
                    ligne = fi.readline().strip()
        else:
            fichier_corpus = kwargs["corpus_filename"]
            self.emb_dim = kwargs["emb_dim"]
            nb_tokens = 1
            with open(fichier_corpus, 'r', encoding="utf-8") as fi:
                for line in fi:
                    tokens = line.rstrip().split(" ")
                    for token in tokens:
                        if token not in self.dico_voca:
                            self.word_array.append(token)
                            self.dico_voca[token] = nb_tokens
                            nb_tokens += 1
            self.vocab_size = nb_tokens
            print("vocab size =", self.vocab_size, "emb_dim =", self.emb_dim)
            self.matrice = torch.zeros((self.vocab_size, self.emb_dim))

    def get_word_index(self, mot):
        return self.dico_voca.get(mot, self.dico_voca['<unk>'])

    def get_emb(self, mot):
        index = self.get_word_index(mot)
        return self.matrice[index]

    def get_emb_torch(self, indice_mot):
        return self.matrice[indice_mot]

    def get_one_hot(self, mot):
        vect = torch.zeros(len(self.dico_voca))
        vect[self.get_word_index(mot)] = 1
        return vect

    def get_word(self, index):
        return self.word_array[index] if index < len(self.word_array) else None




