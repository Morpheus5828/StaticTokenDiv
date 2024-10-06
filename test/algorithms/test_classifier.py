from unittest import TestCase
import time
import numpy as np
import static_token_div.algorithms.classifier as classifer

test_vocab = "test_vocab"
test_example = "test_example"


class TestClassifier(TestCase):
    def test_read_vocab(self):
        result = classifer.read_vocab(test_vocab)
        expected = np.array([1, 2, 3])
        self.assertTrue(np.all(result) == np.all(expected))

    def test_read_example(self):
        result = classifer.read_example(test_example)
        print(result)
        #TODO Finir le test

    def test_create_matrices(self):
        example = classifer.read_example(test_example)
        C, W = classifer.create_matrices(example, vocab_size=3, embedding_size=6)
        print(C.shape, W.shape)

    def test_update(self):
        example = classifer.read_example(test_example)
        C, W = classifer.create_matrices(example, vocab_size=3, embedding_size=9)
        losses = classifer.update(W, C, cpos_index=3, cneg_index=3)
        #print(C.shape, W.shape)
        #print(losses)










