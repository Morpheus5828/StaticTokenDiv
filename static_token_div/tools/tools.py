from typing import Dict, Any, Tuple, List

import numpy as np
from tqdm import tqdm
import random
import pandas as pd


def sigmoid(
        z: np.ndarray,
) -> float:
    return 1 / (1 + np.exp(-z))


def get_text(text_path: str) -> list[str]:
    text_path = text_path.lower()
    with open(text_path, "r", encoding='utf-8') as vocab_file:
        text = vocab_file.read()
    return text.split()


def create_vocabulary(text: list) -> dict[str, int]:
    vocab = []
    seen = set()
    for word in text:
        if word not in seen:
            seen.add(word)
            vocab.append(word)
    number_vocab = {}
    i = 0
    for unique_word in vocab:
        number_vocab[unique_word] = i
        i = i + 1
    return number_vocab


def get_occurrences(
        text: list,
        vocab: dict,
        minc: int
) -> set:
    occurences = set()
    for word in vocab:
        if text.count(word) >= minc:
            occurences.add(word)

    return occurences


def break_list_for_txt(embedding: list) -> str:
    result = ""
    for index, word in enumerate(embedding):
        if index == len(embedding) - 1:
            result += str(word)
        else:
            result += str(word) + " "
    return result


def create_embeddings(
        text: list,
        vocab: dict,
        occurences: set,
        L: int,
        word_except: list
) -> list:
    all_embedding = []
    # for sentence in all_text:

    index = 0
    for word in text:
        if word not in word_except and word in occurences:
            current_embedding = []
            for i in range(-L, L + 1):
                try:
                    current_embedding.append(vocab.get(text[index + i]))
                except:
                    print(f"Error detected when created embedding : " + word + " and : " + text[index + i])
            all_embedding.append(current_embedding)
        index = index + 1
    return all_embedding


def create_pos_context(
        embeddings: list,
        vocab: dict,
        occurrences: set,
        L: int,
        word_except: list
) -> dict[int | set]:
    pos_context = {}
    for unique_word in vocab:
        if unique_word in occurrences and unique_word not in word_except:
            pos_context[vocab.get(unique_word)] = set()
    for embedding in embeddings:
        for i in range(L):
            pos_context.get(embedding[2]).add(embedding[i])
        for i in range(L + 1, 2 * L + 1):
            pos_context.get(embedding[2]).add(embedding[i])
    return pos_context


def create_neg_context(
        pos_context: dict[int | set],
        vocab: dict,
        k: int,
        occurrences: set,
        word_except: list
) -> dict[int | set]:
    neg_context = {}
    for unique_word in vocab:
        if unique_word in occurrences and unique_word not in word_except:
            neg_context[vocab.get(unique_word)] = set()
    for word in pos_context.keys():
        allowed_list = list(vocab.values())
        for word_to_remove in pos_context.get(word):
            if word_to_remove in allowed_list:
                allowed_list.remove(word_to_remove)
        for i in range(k * len(pos_context.get(word))):
            if allowed_list:
                random_word = random.choice(allowed_list)
                neg_context.get(word).add(random_word)
                allowed_list.remove(random_word)
            else:
                break

    return neg_context


def generate_vocab_file(
        vocab: dict[str | int],
        occurrences: set,
        word_except: list,
        save_path: str
):
    to_save = ""
    for main_word in vocab.keys():
        if main_word in occurrences and main_word not in word_except:
            to_save += str(vocab.get(main_word)) + "\n"

    with open(save_path, "w", encoding='utf-8') as f:
        f.write(to_save)


def generate_embeddings_file(
        text: list,
        save_path: str,
        word_except: list,
):
    vocab = create_vocabulary(text)

    occurrences = get_occurrences(text, vocab, 1)
    embeddings = create_embeddings(text, vocab, occurrences, 2, word_except)
    pos_context = create_pos_context(embeddings, vocab, occurrences, 2, word_except)
    neg_context = create_neg_context(pos_context, vocab, 1, occurrences, word_except)

    to_save = ""

    for main_word in pos_context.keys():
        for pos_word in pos_context.get(main_word):
            to_save += str(main_word) + " " + str(pos_word) + " 1\n"
        for neg_word in neg_context.get(main_word):
            to_save += str(main_word) + " " + str(neg_word) + " 0\n"

    print(f"\tFile contains: {len(embeddings)} lines")
    with open(save_path, "w", encoding='utf-8') as f2:
        f2.write(to_save)




def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def extract_embeddings_data(
        txt_path: str,
):
    df = pd.read_csv(txt_path, delimiter=' ')
    return df
