import numpy as np


def sigmoid(
    m: np.ndarray,
    c: np.ndarray,
    prob: str
) -> float:
    if prob == "neg":
        return 1 / (1 + np.exp(np.multiply(-m, c)))
    elif prob == "pos":
        return 1 / (1 + np.exp(np.multiply(m, c)))
    else:
        print("prob not recognized")
        return 0


def get_word_occurrence(
    text: list,
    word: str
) -> int:
    return text.count(word)


def get_text(text_path: str) -> list:
    text = []
    with open(text_path, "r", encoding='utf-8') as vocab_file:
        for line in vocab_file:
            tab_words = line.split(" ")
            # remove indent symbol
            tab_words = list(map(lambda x: x.replace('</s>\n', '</s>'), tab_words))
            # extract all word and add them into text list
            for word in tab_words:
                text.append(word)
    return text


def create_vocabulary(all_text: list) -> set:
    vocab = set()
    for word in all_text:
        vocab.add(word)
    return vocab


def embedding_sentence(
    L: int,
    minc: int,
    text: list
) -> list:


    pass


    # all_embedding = []
    # for i in range(L, len(sentence)-L-1):
    #     if minc is not None and text is not None:
    #         if get_word_occurrence(word=sentence[i], text=text) >= minc:
    #             current_embedding = []
    #             for j in range(-L, L+1):
    #                 current_embedding.append(sentence[i+j])
    #             all_embedding.append(current_embedding)
    #     else:
    #         current_embedding = []
    #         for j in range(-L, L+1):
    #             current_embedding.append(sentence[i+j])
    #         all_embedding.append(current_embedding)
    #
    # return all_embedding
















