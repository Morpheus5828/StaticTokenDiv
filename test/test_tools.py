from unittest import TestCase
from static_token_div.tools import tools
import time
import tqdm

class TestTools(TestCase):
    def test_embedding_sentence(self):
        L = 2

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