from unittest import TestCase
import static_token_div.tools.w2v_tools as w2v_tools
import static_token_div.tools.tools as tools

import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

text_path = os.path.join(project_path, "resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok")



class TestW2vTools(TestCase):
    def setUp(self):
        self.sentence = ['<s>', '<s>', 'Ce', 'chat', 'aime', 'un', 'autre', 'chat', '</s>', '</s>']
        self.word_except = ['<s>', '</s>']

    def test_create_embeddings(self):
        print("TEST test_learning_file")
        vocab = w2v_tools.create_vocabulary(self.sentence, minc=1, word_except=self.word_except)
        occurrences = w2v_tools._get_occurrences(self.sentence, vocab, 1)
        w2v_tools._create_embeddings(
            text=self.sentence,
            vocab=vocab,
            occurrences=occurrences,
            L=2,
            word_except=self.word_except
        )

        with open("../test_text/test", "r") as file:
            lines = file.readlines()
            self.assertTrue(len(lines) > 0, "Le fichier d'embeddings doit contenir des données")

    def test_pos_context(self):
        print("TEST test_pos_context")

        vocab = w2v_tools.create_vocabulary(self.sentence, minc=1, word_except=self.word_except)
        occurrences = w2v_tools._get_occurrences(self.sentence, vocab, 1)
        embeddings = w2v_tools._create_embeddings(self.sentence, vocab, occurrences, 2, self.word_except)
        pos_context = w2v_tools._create_pos_context(embeddings, vocab, occurrences, 2, self.word_except)

        self.assertTrue(len(pos_context.get(vocab.get('chat'))) == 5)

    def test_neg_context(self):
        print("TEST test_neg_context")

        vocab = w2v_tools.create_vocabulary(self.sentence, minc=1, word_except=self.word_except)
        occurrences = w2v_tools._get_occurrences(self.sentence, vocab, 1)
        embeddings = w2v_tools._create_embeddings(self.sentence, vocab, occurrences, 2, self.word_except)
        pos_context = w2v_tools._create_pos_context(embeddings, vocab, occurrences, 2, self.word_except)
        neg_context = w2v_tools._create_neg_context(pos_context, vocab, 1, occurrences, self.word_except)

        self.assertTrue(len(neg_context) == 5)
        self.assertTrue(len(neg_context.get(vocab.get('chat'))) == 1)

    def test_get_frequencies(self):
        text = tools.get_text(text_path)
        vocab = w2v_tools.create_vocabulary(text, minc=4, word_except=self.word_except)
        result = w2v_tools.random_choice_using_prob(vocab, k=3)
        self.assertTrue(np.unique(result).shape[0] == result.shape[0])