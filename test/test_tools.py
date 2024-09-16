from unittest import TestCase
from static_token_div import tools


class TestTools(TestCase):
    def test_embedding_sentence(self):
        sentence = "The fox is runing into the forest."
        sentence = sentence.split(" ")
        n = 5
        L = 2
        k = 2
        result = tools.embedding_sentence(
            sentence,
            n=n,
            L=L,
            k=k
        )

