from unittest import TestCase
import static_token_div.tools.tools as tools
import numpy as np


class TestTools(TestCase):
    def test_get_text(self):
        print("TEST test_create_text")
        text_path = "../../resources/tlnl_tp1_data/alexandre_dumas/Le_comte_de_Monte_Cristo.tok"
        text = tools.get_text(text_path)
        self.assertTrue(len(text) == 9512)

    def test_sigmoid(self):
        print("TEST test_sigmoid")
        input_data = np.array([0, 2, -2])
        expected_output = np.array([0.5, 0.88079708, 0.11920292])
        result = tools.sigmoid(input_data)
        np.testing.assert_almost_equal(result, expected_output, decimal=7)

    def test_safe_log(self):
        print("TEST test_safe_log")
        input_data = np.array([1.0, 0.1, 1e-12, 0.0])
        expected_output = np.log(np.clip(input_data, 1e-10, 1.0))
        result = tools._safe_log(input_data)
        np.testing.assert_almost_equal(result, expected_output, decimal=10)

    def test_loss_function(self):
        print("TEST test_loss_function")
        m = np.array([0.5, -0.2])
        c_pos = np.array([0.4, 0.6])
        c_neg = np.array([[-0.1, 0.7], [0.8, -0.5]])

        pos_loss = tools._safe_log(tools.sigmoid(np.dot(m, c_pos)))
        neg_loss = np.sum(np.log(tools.sigmoid(-np.dot(m, c_neg.T))))
        expected_loss = -(pos_loss + neg_loss)

        result = tools.loss_function(m, c_pos, c_neg)
        self.assertAlmostEqual(result, expected_loss, places=6)

    def test_cosine_similarity(self):
        print("TEST test_cosine_similarity")
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        expected_output = 0.0
        result = tools.cosine_similarity(v1, v2)
        self.assertAlmostEqual(result, expected_output, places=6)

        v3 = np.array([1, 2, 3])
        v4 = np.array([1, 2, 3])
        expected_output = 1.0
        result = tools.cosine_similarity(v3, v4)
        self.assertAlmostEqual(result, expected_output, places=6)





