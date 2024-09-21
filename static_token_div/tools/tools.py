from typing import Dict, Any

import numpy as np
from tqdm import tqdm

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


def get_text_word_occurence(
    all_text: list,
    vocab: dict,
    minc: int
) -> dict:
    occurences = set()
    for word in vocab:
        if get_word_occurrence(all_text, word) >= 5:
            occurences.add(word)

    return occurences


def filter_occurence(
    occurencies: dict,
    step: int
) -> dict:
    return {word: occurence for word, occurence in occurencies.items() if occurence > step}


def get_text(text_path: str) -> list:
    text_path = text_path.lower()
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


def break_list_for_txt(embedding: list) -> str:
    result = ""
    for index, word in enumerate(embedding):
        if index == len(embedding) - 1:
            result += str(word)
        else:
            result += str(word) + " "
    return result


def create_vocabulary(all_text: list) -> dict[Any, int | Any]:
    vocab = []
    seen = set()
    for word in all_text:
        if word not in seen:
            seen.add(word)
            vocab.append(word)
    number_vocab = {}
    i = 0
    for unique_word in vocab:
        number_vocab[unique_word] = i
        i = i + 1
    return number_vocab


def get_embedding_sentence(
    L: int,
    minc: int,
    text_path: str,
    word_except: list = ["<s>", "</s>"]
) -> list:
    # create vocabulary from text
    all_text = get_text(text_path)
    vocab = create_vocabulary(all_text)
    # compute all words occurence in text
    occurencies = get_text_word_occurence(all_text, vocab)
    # create embedding from main occurencies words
    all_embedding = []
    #for sentence in all_text:

    for current_word in occurencies.keys():
        for word_index in range(len(all_text)):
            if current_word == all_text[word_index] and (current_word not in word_except):
                current_embedding = []
                for i in range(-L, L+1):
                    try:
                        current_embedding.append(vocab.get(all_text[word_index+i]))
                    except:
                        print(f"Error detected when created embedding")
                all_embedding.append(current_embedding)
    return all_embedding
















