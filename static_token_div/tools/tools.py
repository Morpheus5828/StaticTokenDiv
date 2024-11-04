""" This script contains different function using in Word2Vec algorithm
.module author:: Marius THORRE
"""

import numpy as np
import math


def get_text(text_path: str) -> list[str]:
    text_path = text_path.lower()
    with open(text_path, "r", encoding='utf-8') as vocab_file:
        text = vocab_file.read().split("\n")
    return text


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def _safe_log(x: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    return np.log(np.clip(x, epsilon, 1.0))


def loss_function(m: np.ndarray, c_pos: np.ndarray, c_neg: np.ndarray) -> np.ndarray:
    pos_loss = _safe_log(sigmoid(np.dot(m, c_pos)))
    neg_loss = np.sum(np.log(sigmoid(-np.dot(m, c_neg.T))))
    return - (pos_loss + neg_loss)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_prob(frequencies: float, t: float) -> float:
    return 1 - math.sqrt(t/(frequencies+1))

