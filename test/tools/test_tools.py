from unittest import TestCase
from static_token_div.tools import tools
import time


class TestTools(TestCase):

    def test(self):
        test = {"a": [1,2,3], "b": [2,3,4]}
        for word in test.get("a"):
            print(word)
    def test_create_text(self):
        print("TEST test_create_text")
        text_path = "../../resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok"
        text = tools.get_text(text_path)
        print(text[0])


    def test_embedding_sentence(self):
        print("TEST test_embedding_sentence")
        start = time.time()
        print("\tStarting embedding creation extraction process ...")

        text_path = "../../resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok"
        L=2
        minc=5

        result = tools.create_embeddings(
            L=L,
            minc=minc,
            text_path=text_path
        )
        end = time.time()

        print(f"\tVocab creation process time: {end - start:.2f} s")

        print(result[0])
        #self.assertTrue(result[0] == [6226, 7904, 10220, 10472, 3628])
        #self.assertTrue(len(result) == 184581) #nb of embedding available

    def test_create_vocabulary(self):
        print("TEST test_create_vocabulary")
        text_path = "../../resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok"
        text = tools.get_text(text_path)
        vocab = tools.create_vocabulary(text)
        print(vocab["</s>"])

    def test_get_word_occurrence(self):
        print("TEST test_get_word_occurrence")
        text_path = "../../resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok"
        text = tools.get_text(text_path)
        vocab = tools.create_vocabulary(text)
        occurences = {}
        start = time.time()
        print("\tStarting vocab extraction process ...")
        for word in vocab:
            occurences[word] = tools.get_word_occurrence(word=word, text=text)
        end = time.time()
        print(f"\tVocab creation process time: {end - start:.2f} s")
        occurences = dict(sorted(occurences.items(), key=lambda item: item[1], reverse=True))

        print(occurences["."])
        self.assertTrue(occurences["<s>"] == 19027)
        self.assertTrue(occurences["</s>"] == 19024)

    def test_break_list_for_txt(self):
        sentence = ['février', '1815', ',', 'la', 'vigie']

        result = tools.break_list_for_txt(sentence)
        self.assertTrue(result == "février 1815 , la vigie")

