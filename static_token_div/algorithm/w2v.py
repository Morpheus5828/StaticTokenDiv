""" This script is an embedding file generator
.module author:: Marius THORRE
"""


def embedding_generator(
    text: str,
    n: int,
    L: int,
    k: int,
    eta: float,
    e: int,
    minc: int

) -> None:
    """
    Embedding generator file

    :param text: text to apply embeddings
    :param n: embeddings dimensions
    :param L: right and left context size, windows size is equal to 2*L + 1
    :param k: negative nb context for a positive context
    :param eta: learning rate
    :param e: nb iterations
    :param minc: minimal occurrence number 
    :return: None
    """
    text_split = text.split(" ") # split text by space between word
    text_embedding = []
    for i in range(1, len(text_split)):
        print("target: ", text_split[i])
        for j in range(-L, L+1):
            if i != 1 or i == len(text_split) -2:
                print(text_split[i+j], end=" ")
        print()


def classifier(
    first_word: str,
    second_word: str,

) -> float:
    """

    :param first_word:
    :param second_word:
    :return:
    """


