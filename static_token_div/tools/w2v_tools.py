import numpy as np
import random
from static_token_div.tools.tools import get_text
from static_token_div.tools.embedding_tools import create_vocabulary

def vocab_idx(vocab):
    new_vocab = {}
    for word in vocab.keys():
        new_vocab[vocab[word][0]] = word
    return new_vocab


def create_learning_file(
    text_path: str,
    L: int,
    k: int,
    word_except: list,
    minc: int,
    optmim_random_choice: bool = False
) -> tuple:
    """
    This script creates a learning file for the Word2Vec algorithm.
    """
    text = get_text(text_path)
    vocab, word_counts_in_vocab, total_word_count = create_vocabulary(text, minc, word_except)
    vocab_list = list(vocab.keys())

    if optmim_random_choice:
        word_counts_array = np.array([word_counts_in_vocab[word] for word in vocab_list], dtype=np.float64)
        word_freqs = word_counts_array / np.sum(word_counts_array)
        adjusted_freqs = word_freqs ** (3 / 4)
        probabilities = adjusted_freqs / np.sum(adjusted_freqs)
    else:
        probabilities = None

    training_data = []
    for sentence in text:
        words = sentence.split()
        for i, word in enumerate(words):
            if word not in vocab:
                continue
            target_word = vocab[word]
            # Positive context words
            pos_context_indices = list(range(max(0, i - L), min(len(words), i + L + 1)))
            pos_context_indices.remove(i)
            for j in pos_context_indices:
                context_word = words[j]
                if context_word in vocab:
                    context_word_idx = vocab[context_word]
                    # Negative sampling
                    c_neg = []
                    for _ in range(k):
                        if optmim_random_choice:
                            neg_word = np.random.choice(vocab_list, p=probabilities)
                        else:
                            neg_word = random.choice(vocab_list)
                        while neg_word == word or neg_word in word_except:
                            if optmim_random_choice:
                                neg_word = np.random.choice(vocab_list, p=probabilities)
                            else:
                                neg_word = random.choice(vocab_list)
                        neg_context_word = vocab[neg_word]
                        c_neg.append(neg_context_word[0])
                    training_data.append(np.array([target_word[0], context_word_idx[0], *c_neg]))

    return training_data, vocab_idx(vocab)


def _get_occurrences(
    text: list,
    vocab: dict,
    minc: int
) -> set:
    """
    Getter to have occurrence of each word
    :param text: list of word
    :param vocab: vocab dict
    :param minc: minimal occurrence word
    :return: set with occurrences
    """

    occurrences = set()
    for word in vocab:
        if text.count(word) >= minc:
            occurrences.add(word)

    return occurrences


def _create_embeddings(
    text: list,
    vocab: dict,
    occurrences: set,
    L: int,
    word_except: list
) -> list:
    """
    This file create embedding
    :param text: text string
    :param vocab: vocab from text
    :param occurrences: set occurrences
    :param L: size of embedding
    :param word_except: list of word exceptions
    :return: list of embeddings
    """
    all_embedding = []

    index = 0
    for word in text:
        if word not in word_except and word in occurrences:
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


def _create_pos_context(
    embeddings: list,
    vocab: dict,
    occurrences: set,
    L: int,
    word_except: list
) -> dict[int | set]:
    """
    Create positive context
    :param embeddings: list of embeddings
    :param vocab: vocab from text
    :param occurrences: set of occurrences
    :param L: embedding size
    :param word_except: list of word exceptions
    :return: dict <key, value> key is word and value his context
    """
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


def _create_neg_context(
    pos_context: dict[int | set],
    vocab: dict,
    k: int,
    occurrences: set,
    word_except: list
) -> dict[int | set]:
    """
    Create negative context
    :param pos_context: dict of positive context
    :param vocab: vocab from text
    :param k: number of c_neg to generate
    :param occurrences: set of occurrences
    :param word_except: list of word exceptions
    :return: dict <key, value> key is word and value set of c_neg
    """
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

    return neg_context



