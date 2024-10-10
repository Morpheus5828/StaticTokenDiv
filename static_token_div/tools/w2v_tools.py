import numpy as np
import random
from collections import defaultdict
from static_token_div.tools.tools import get_text, sigmoid


# TODO marche pas
def generate_embeddings_file(
        text_path: str,
        save_path: str,
        word_except: list,
        L: int,
        k: int,
        minc: int = 1
):
    text = get_text(text_path)
    vocab = _create_vocabulary(text)
    print(f"\tInitial vocab size: {len(vocab)}")
    occurrences = _get_occurrences(text, vocab, minc)
    print(f"\tNew vocab size: {len(occurrences)} after skip minc")
    embeddings = _create_embeddings(text, vocab, occurrences, L, word_except)
    pos_context = _create_pos_context(embeddings, vocab, occurrences, L, word_except)
    neg_context = _create_neg_context(pos_context, vocab, k, occurrences, word_except)
    to_save = ""
    print(f"\tNb lines: {len(pos_context) * len(neg_context) * len(occurrences)}")
    print("\tStart writing file ...")
    for main_word in pos_context.keys():
        for pos_word in pos_context.get(main_word):
            for neg_word in neg_context.get(main_word):
                to_save += str(main_word) + " " + str(pos_word) + str(neg_word)

    print(f"\tFile contains: {len(embeddings)} lines")
    with open(save_path, "w", encoding='utf-8') as f2:
        f2.write("coucou")


def _create_learning_file(
        text: list,
        L: int,
        k: int,
        word_except: list,
        minc: int
) -> list:
    vocab = _create_vocabulary(text, minc, word_except)
    training_data = []
    vocab_list = list(vocab.keys())
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
                if words[j] in vocab:
                    context_word_idx = vocab[words[j]]
                    # Negative sampling
                    c_neg = []
                    for _ in range(k):
                        neg_word = random.choice(vocab_list)
                        while neg_word == word or neg_word in word_except:
                            neg_word = random.choice(vocab_list)
                        neg_context_word = vocab[neg_word]
                        c_neg.append(neg_context_word[0])

                    training_data.append((target_word[0], context_word_idx[0], *c_neg))

    return training_data


def get_embedding(
        text_path: str,
        word_except: list,
        L: int,
        k: int,
        minc: int = 1
):
    text = get_text(text_path)
    vocab = _create_vocabulary(text)
    occurrences = _get_occurrences(text, vocab, minc)
    embeddings = _create_embeddings(text, vocab, occurrences, L, word_except)
    pos_context = _create_pos_context(embeddings, vocab, occurrences, L, word_except)
    neg_context = _create_neg_context(pos_context, vocab, k, occurrences, word_except)
    result = []
    target = [line[2] for line in embeddings]
    for element in vocab.keys():
        print(vocab.get(element)[0])


'''

    neg_context = _create_neg_context(pos_context, vocab, k, occurrences, word_except)
    embedding = {}
    for main_word in pos_context.keys():
        context = {}
        for pos_word in pos_context.get(main_word):
            c_neg = [neg_word for neg_word in neg_context.get(main_word)]
            context["c_pos"] = pos_word
            context["c_neg"] = c_neg'''


def _create_vocabulary(
        text: list,
        minc: int,
        word_except: list
) -> dict:
    word_counts = defaultdict(int)
    for sentence in text:
        for word in sentence.split():
            word_counts[word] += 1
        vocab = {word: (idx, count) for idx, (word, count) in enumerate(word_counts.items()) if count >= minc and word not in word_except}
        reindexed_vocab = {word: (new_idx, count) for new_idx, (word, (_, count)) in enumerate(vocab.items())}
    return reindexed_vocab


def _get_occurrences(
        text: list,
        vocab: dict,
        minc: int
) -> set:
    occurences = set()
    for word in vocab:
        if text.count(word) >= minc:
            occurences.add(word)

    return occurences


def _create_embeddings(
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


def _create_pos_context(
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


def _create_neg_context(
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

    return neg_context


def _get_c_neg(
        vocab: dict,
        k: int,
        occurrences: set,
        word_except: list
):
    neg_context = []
