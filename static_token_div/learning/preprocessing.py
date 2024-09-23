""" This script is a pre-processing file for classifier
.module author:: Marius THORRE
"""
import random
from static_token_div.algorithms import w2v
import numpy as np


def extract_context(
        context_file_path: str,
) -> tuple:
    """
    Pre processing file
    :param context_file_path:
    :return:
    """
    # extract context
    with open(context_file_path, "r", encoding='utf-8') as file:
        positive_context = []
        negative_context = []
        step = 0
        for line in file:
            if line == "-----separation------\n":
                step = 1
                continue

            line = line.split(" ")
            line = [int(word) for word in line]
            if step == 0:
                positive_context.append(np.array(line))
            else:
                negative_context.append(np.array(line))
    # convert to numpy array
    positive_context = np.array(positive_context)
    negative_context = np.array(negative_context)

    return positive_context, negative_context


def create_learning_data(
        positive_context: np.ndarray,
        negative_context: np.ndarray
) -> tuple:
    input_data = np.concatenate((positive_context, negative_context))
    target_data = np.concatenate((np.zeros(positive_context.shape[0]), np.ones(negative_context.shape[0])))

    return input_data, target_data




