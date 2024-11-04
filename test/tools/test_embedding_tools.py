from unittest import TestCase
import static_token_div.tools.embedding_tools as embedding_tools


class TestEmbeddingTools(TestCase):
    def setUp(self):
        self.sentence = ['<s>', '<s>', 'Ce', 'chat', 'aime', 'un', 'autre', 'chat', '</s>', '</s>']
        self.word_except = ['<s>', '</s>']
        self.vocab = embedding_tools._create_vocabulary(self.sentence)
        self.context = embedding_tools._create_context(self.sentence, self.vocab, 2, 1, self.word_except, 1)
        self.saving_path = "test.txt"

    def test_create_context(self):
        for target, context_word, label in self.context:
            self.assertIn(label, [0, 1])
            self.assertTrue(target in self.vocab.values())
            self.assertTrue(context_word in self.vocab.values())

    def test_create_vocabulary(self):
        vocab = embedding_tools._create_vocabulary(self.sentence)
        self.assertTrue(len(vocab) > 0)
        for word, (index, count) in vocab.items():
            self.assertTrue(isinstance(index, int))
            self.assertTrue(isinstance(count, int))
            self.assertTrue(count >= 1)
