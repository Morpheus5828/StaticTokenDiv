""" This script contains different function for create embedding
.module author:: Marius THORRE
"""

import random
from collections import defaultdict
from typing import Set

from static_token_div.tools.tools import get_text


def get_embedding_from_trigram(vocab: dict, embedding: dict, word: int):
    current_word = ""
    for mot, (id_val, ct) in vocab.items():
        if id_val == word:
            current_word = mot
            break
    words = create_n_gram(current_word, n=3)
    result = sum([embedding[vocab[w][0]] for w in words])
    return result






def embedding_generator(
        saving_path: str,
        text_path: str,
        L: int,
        k: int,
        minc: int,
        word_except: list,

) -> None:
    """
    Embedding generator file
    :param saving_path: embedding saving path file
    :param text_path: text to apply embeddings
    :param L: right and left context size, windows size is equal to 2*L + 1
    :param k: negative nb context for a positive context
    :param minc: minimal occurrence number
    :param word_except: word except, will not appear as embedding target
    :return: None
    """

    text = get_text(text_path)
    vocab = _create_vocabulary(text)
    print(f"\tVocab size: {len(vocab)}")
    context = _create_context(text, vocab, L, k, word_except, minc)
    _generate_training_data(context, saving_path)


def _create_context(
        text: list,
        vocab: dict,
        L: int,
        k: int,
        word_except: list,
        minc: int
) -> list:
    """
    This function create positive and negative context from target
    :param text: list of sentences
    :param vocab: vocab of current text
    :param L: size oh embedding
    :param k: nb of negative context
    :param word_except: word except won't be appear in process
    :param minc: minimal occurrences step for each word in text
    :return: list of contexts
    """
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


def _generate_training_data(
        context: list,
        training_path: str,

) -> None:
    """
    Generate training file
    :param context: list of context
    :param training_path: string of training path
    :return: None
    """
    to_save = ""
    for data in context:
        to_save += str(data[0]) + " " + str(data[1]) + " " + str(data[2]) + "\n"

    with open(training_path, "w", encoding='utf-8') as f:
        f.write(to_save)


def _create_vocabulary(
        text: list
) -> dict:
    """
    Function which create vocabulary
    :param text: list of string which contains text
    :return: dict <key, value> where key is word in vocab and value his div
    """
    vocab = {}
    word_counts = defaultdict(int)
    for sentence in text:
        for word in sentence.split():
            word_counts[word] += 1
        vocab = {word: (idx, count) for idx, (word, count) in enumerate(word_counts.items())}
    return vocab


def create_n_gram(word: str, n: int) -> set[str]:
    result = []
    padded_word = '<' + word + '>' if n > 1 else word
    for i in range(len(padded_word) - n + 1):
        n_gram = padded_word[i:i+n]
        result.append(n_gram)
    return set(result)


def create_vocabulary(
    text: list,
    minc: int,
    word_except: list,
    add_n_gram: bool = False,
    n: int = 3,
):
    """
    Create vocabulary from a text
    :param text: list of string
    :param minc: minimal occurrence word
    :param word_except: list of word exceptions
    :param add_n_gram: condition to add n-gram word
    :param n: length of n-grams
    :return: vocab with only words that meet minc and word_except conditions,
             word_counts_in_vocab, total_word_count
    """
    word_counts = defaultdict(int)

    for sentence in text:
        for word in sentence.split():
            if word not in word_except:
                word_counts[word] += 1
                if add_n_gram and len(word) >= n:
                    for n_gram in create_n_gram(word, n):
                        word_counts[n_gram] += 1

    vocab = {word: (idx, count) for idx, (word, count) in enumerate(word_counts.items())
             if count >= minc and word not in word_except}
    word_counts_in_vocab = {word: count for word, (idx, count) in vocab.items()}
    total_word_count = sum(word_counts.values())
    return vocab, word_counts_in_vocab, total_word_count


