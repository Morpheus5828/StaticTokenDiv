from unittest import TestCase
from static_token_div.tools import tools
import time


class TestTools(TestCase):
    def test_create_text(self):
        print("TEST test_create_text")
        text_path = "../../resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok"
        text = tools.get_text(text_path)
        print(text[0])

    def test_create_vocabulary(self):
        print("TEST test_create_vocabulary")
        sentence = ['<s>', '<s>', 'Ce', 'chat', 'aime', 'un', 'autre', 'chat', '</s>', '</s>']
        vocab = tools.create_vocabulary(sentence)
        self.assertTrue(vocab["<s>"] == 0)
        self.assertTrue(vocab["</s>"] == 6)
        self.assertTrue(vocab["chat"] == 2)

    def test_get_word_occurrence(self):
        print("TEST test_get_word_occurrence")
        sentence = ['<s>', '<s>', 'Ce', 'chat', 'aime', 'un', 'autre', 'chat', '</s>', '</s>']
        vocab = tools.create_vocabulary(sentence)
        occurrences = tools.get_occurrences(sentence, vocab, 1)

        self.assertTrue("chat" in occurrences)
        self.assertTrue("un" in occurrences)

        occurrences = tools.get_occurrences(sentence, vocab, 2)

        self.assertTrue("chat" in occurrences)
        self.assertTrue("un" not in occurrences)

    def test_embedding_sentence(self):
        print("TEST test_embedding_sentence")
        print("\tStarting embedding creation extraction process ...")

        print("TEST test_get_word_occurrence")
        sentence = ['<s>', '<s>', 'Ce', 'chat', 'aime', 'un', 'autre', 'chat', '</s>', '</s>']
        vocab = tools.create_vocabulary(sentence)
        occurrences = tools.get_occurrences(sentence, vocab, 1)
        embeddings = tools.create_embeddings(sentence, vocab, occurrences, 2, word_except=['<s>', '</s>'])

        self.assertTrue(embeddings[0] == [0, 0, 1, 2, 3])
        self.assertTrue(embeddings[-1] == [4, 5, 2, 6, 6])

    def test_break_list_for_txt(self):
        sentence = ['février', '1815', ',', 'la', 'vigie']

        result = tools.break_list_for_txt(sentence)
        self.assertTrue(result == "février 1815 , la vigie")

    def test_pos_context(self):
        sentence = ['<s>', '<s>', 'Ce', 'chat', 'aime', 'un', 'autre', 'chat', '</s>', '</s>']
        word_except = ['<s>', '</s>']
        vocab = tools.create_vocabulary(sentence)
        occurrences = tools.get_occurrences(sentence, vocab, 1)
        embeddings = tools.create_embeddings(sentence, vocab, occurrences, 2, word_except)
        pos_context = tools.create_pos_context(embeddings, vocab, occurrences, 2, word_except)

        self.assertTrue(pos_context.get(1) == {0, 2, 3})
        self.assertTrue(pos_context.get(2) == {0, 1, 3, 4, 5, 6})
        self.assertTrue(pos_context.get(5) == {3, 4, 2, 6})

    def test_neg_context(self):
        sentence = ['<s>', '<s>', 'Ce', 'chat', 'aime', 'un', 'autre', 'chat', '</s>', '</s>']
        word_except = ['<s>', '</s>']
        vocab = tools.create_vocabulary(sentence)
        occurrences = tools.get_occurrences(sentence, vocab, 1)
        embeddings = tools.create_embeddings(sentence, vocab, occurrences, 2, word_except)
        pos_context = tools.create_pos_context(embeddings, vocab, occurrences, 2, word_except)
        neg_context = tools.create_neg_context(pos_context, vocab, 1, occurrences, word_except)

        self.assertTrue(len(neg_context.get(1)) == 3)
        self.assertTrue(len(neg_context.get(2)) == 1)
        self.assertTrue(len(neg_context.get(4)) == 3)

    def test_generate_embeddings_file(self):
        sentence = ['<s>', '<s>', 'Ce', 'chat', 'aime', 'un', 'autre', 'chat', '</s>', '</s>']
        word_except = ['<s>', '</s>']
        # vocab = tools.create_vocabulary(sentence)
        # occurrences = tools.get_occurrences(sentence, vocab, 1)
        # embeddings = tools.create_embeddings(sentence, vocab, occurrences, 2, word_except)
        # pos_context = tools.create_pos_context(embeddings, vocab, occurrences, 2, word_except)
        # neg_context = tools.create_neg_context(pos_context, vocab, 1, occurrences, word_except)
        tools.generate_embeddings_file(sentence, "test.txt", word_except)

    def test_extract(self):
        df = tools.extract_embeddings_data("../../static_token_div/learning/learning_file.txt")
        print(df.columns)