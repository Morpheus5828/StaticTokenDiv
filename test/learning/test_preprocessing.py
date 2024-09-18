from unittest import TestCase
import time
import static_token_div.learning.preprocessing as preprocessing


class TestPreprocessing(TestCase):
    def test_process(self):
        embedding_path = "../embedding_generated.txt"
        k = 2

        start = time.time()

        preprocessing.process(
            embedding_file_path=embedding_path,
            k=k,
        )



        end = time.time()
        print(f"\tTime process: {end - start:.2f} s")