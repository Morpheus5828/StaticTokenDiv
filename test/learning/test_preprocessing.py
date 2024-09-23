from unittest import TestCase
import time
import static_token_div.learning.preprocessing as preprocessing


class TestPreprocessing(TestCase):
    def test_process(self):
        context_path = "../../resources/tlnl_tp1_data/ad_learning/context_generated.txt"

        start = time.time()

        positive_context, negative_context = preprocessing.process(context_file_path=context_path)
        self.assertTrue(positive_context.shape == (229422, 2))
        self.assertTrue(negative_context.shape == (451275, 2))

        end = time.time()
        print(f"\tTime process: {end - start:.2f} s")