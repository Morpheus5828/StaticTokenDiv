from unittest import TestCase
import time
import static_token_div.algorithms.w2v as w2v
from static_token_div.tools.tools import _get_text


class TestW2v(TestCase):
    def test_embedding_generator(self):
        print("TEST test_embedding_generator")

        text_path = "../../resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok"
        save_file_path = "../txt_embedding.txt"
        L = 2
        minc = 10
        start = time.time()
        print("\tStarting embedding generator file creation ...")

        w2v.embedding_generator(
            save_path=save_file_path,
            text_path=text_path,
            L=L,
            minc=minc
        )

        end = time.time()
        print(f"\tCreation process time: {end - start:.2f} s")
        print(f"\tFile save at: {save_file_path}")


