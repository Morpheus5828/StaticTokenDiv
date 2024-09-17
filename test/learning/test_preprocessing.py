from unittest import TestCase
import time
import static_token_div.learning.preprocessing as preprocessing


class TestPreprocessing(TestCase):
    def test_process(self):
        embedding_path = "../txt_embedding.txt"
        k = 5

        preprocessing.process(
            embedding_file_path=embedding_path,
            k=k
        )
