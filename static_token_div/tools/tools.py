from typing import Dict, Any, Tuple, List

import numpy as np
from tqdm import tqdm
import random
import pandas as pd
from collections import defaultdict

def sigmoid(
        z: np.ndarray,
) -> float:
    return 1 / (1 + np.exp(-z))


def get_text(text_path: str) -> list[str]:
    text_path = text_path.lower()
    with open(text_path, "r", encoding='utf-8') as vocab_file:
        text = vocab_file.read().split()
    return text



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

    index = 0
    for word in text:
        if word not in word_except and word in occurences:
            current_embedding = []
            for i in range(-L, L + 1):
                try:
                    if len(text) > index + i:
                        current_embedding.append(vocab.get(text[index + i]))
                except:
                    print(f"Error detected when created embedding : " + word)
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
            if len(embedding) > i:
                pos_context.get(embedding[L]).add(embedding[i])
        for i in range(L + 1, 2 * L + 1):
            if len(embedding) > i:
                pos_context.get(embedding[L]).add(embedding[i])
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
        L: int,
        k: int,
        minc: int = 1
):
    vocab = create_vocabulary(text)

    occurrences = get_occurrences(text, vocab, minc)
    embeddings = create_embeddings(text, vocab, occurrences, L, word_except)
    pos_context = create_pos_context(embeddings, vocab, occurrences, L, word_except)
    neg_context = create_neg_context(pos_context, vocab, k, occurrences, word_except)

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


def get_text2(text_path: str) -> list[str]:
    text_path = text_path.lower()
    with open(text_path, "r", encoding='utf-8') as vocab_file:
        text = vocab_file.read().split("\n")
    return text


def create_vocabulary2(
        text: list
) -> dict:
    word_counts = defaultdict(int)
    for sentence in text:
        for word in sentence.split():
            word_counts[word] += 1
        vocab = {word: (idx, count) for idx, (word, count) in enumerate(word_counts.items())}
    return vocab


def create_context(
        text: list,
        vocab: dict,
        L: int,
        k: int,
        word_except: list,
        minc: int
) -> list:
    training_data = []
    vocab_list = list(vocab.keys())

    for sentence in text:
        words = sentence.split()
        for i, word in enumerate(words):
            if word not in vocab or word in word_except or vocab[words[i]][1] < minc:
                continue
            target_word = vocab[word]
            # Positive context words
            pos_context_indices = list(range(max(0, i - L), min(len(words), i + L + 1)))
            pos_context_indices.remove(i)

            for j in pos_context_indices:
                if words[j] in vocab and words[j] not in word_except:
                    pos_context_word = vocab[words[j]]
                    training_data.append((target_word[0], pos_context_word[0], 1))

                    # Negative sampling
                    for _ in range(k):
                        neg_word = random.choice(vocab_list)
                        while neg_word == word or neg_word in word_except:
                            neg_word = random.choice(vocab_list)
                        neg_context_word = vocab[neg_word]
                        training_data.append((target_word[0], neg_context_word[0], 0))

    return training_data


def generate_training_data(
        context: list,
        training_path: str,

):
    to_save = ""
    for data in context:
        to_save += str(data[0]) + " " + str(data[1]) + " " + str(data[2]) + "\n"

    with open(training_path, "w", encoding='utf-8') as f2:
        f2.write(to_save)