from unittest import TestCase
from static_token_div.tools import tools
import time


class TestTools(TestCase):
    def test_embedding_sentence(self):
        print("TEST test_embedding_sentence")
        start = time.time()
        print("\tStarting embedding creation extraction process ...")

        text_path = "../resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok"
        L=2
        minc=10

        result = tools.get_embedding_sentence(
            L=L,
            minc=minc,
            text_path=text_path
        )
        end = time.time()

        print(f"\tVocab creation process time: {end - start:.2f} s")

        self.assertTrue(result[0] == ['février', '1815', ',', 'la', 'vigie'])
        self.assertTrue(result[1] == ['le', 'pharaon', ',', 'venant', 'de'])
        self.assertTrue(len(result) == 184581) #nb of embedding available

    def test_create_vocabulary(self):
        print("TEST test_create_vocabulary")
        text_path = "../resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok"
        all_text = tools.get_text(text_path)
        vocab = tools.create_vocabulary(all_text)
        print(len(vocab))

    def test_get_word_occurrence(self):
        print("TEST test_get_word_occurrence")
        text_path = "../resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok"
        all_text = tools.get_text(text_path)
        vocab = tools.create_vocabulary(all_text)
        occurences = {}
        start = time.time()
        print("\tStarting vocab extraction process ...")
        for word in vocab:
            occurences[word] = tools.get_word_occurrence(word=word, text=all_text)
        end = time.time()
        print(f"\tVocab creation process time: {end-start:.2f} s")
        occurences = dict(sorted(occurences.items(), key=lambda item: item[1], reverse=True))

        self.assertTrue(occurences["<s>"] == 19027)
        self.assertTrue(occurences["</s>"] == 19024)

    def test_break_list_for_txt(self):
        sentence = ['février', '1815', ',', 'la', 'vigie']

        result = tools.break_list_for_txt(sentence)
        self.assertTrue(result == "février 1815 , la vigie")