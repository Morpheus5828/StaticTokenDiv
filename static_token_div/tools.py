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
    text: str,
    word: str
) -> int:
    return text.count(word)


def embedding_sentence(
    sentence: list,
    n: int,
    L: int,
    k: int,
) -> list:
    empty_line = ["" for _ in range(n)]
    all_embedding = [empty_line for _ in range(len(sentence)-2)]
    target_index = 1
    for i in range(len(sentence)-2):
        for j in range(-L, L):
            if target_index + j > 0:
                #print(target_index+)
                all_embedding[i].append(sentence[target_index + j])
                print(all_embedding)
                print("####")
                #print(sentence[target_index + j], end=" ")
        target_index += 1

    return all_embedding
















