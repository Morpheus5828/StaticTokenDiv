
import random
import numpy as np
from static_token_div.tools.tools import get_text
from static_token_div.tools.embedding_tools import create_vocabulary


def create_learning_file_with_n_gram(
        text_path: str,
        L: int,
        k: int,
        n: int,
        word_except: list,
        minc: int,
        optmim_random_choice: bool = False
):

    text = get_text(text_path)
    vocab, word_counts_in_vocab, total_word_count = create_vocabulary(
        text=text,
        minc=minc,
        word_except=word_except,
        add_n_gram=True,
        n=n
    )

    word_to_index = {word: idx for word, (idx, count) in vocab.items()}
    vocab_list = list(vocab.keys())

    if optmim_random_choice:
        word_counts = np.array([count for word, (idx, count) in vocab.items()], dtype=np.float64)
        word_freqs = word_counts / np.sum(word_counts)
        adjusted_freqs = word_freqs ** 0.75
        probabilities = adjusted_freqs / np.sum(adjusted_freqs)
    else:
        probabilities = None

    training_data = []
    for sentence in text:
        words = sentence.split()
        for i, word in enumerate(words):
            if word not in vocab:
                continue
            target_idx = word_to_index[word]
            context_indices = []

            window_start = max(0, i - L)
            window_end = min(len(words), i + L + 1)
            for j in range(window_start, window_end):
                if j == i:
                    continue
                context_word = words[j]
                if context_word in vocab:
                    context_word_idx = word_to_index[context_word]
                    context_indices.append(context_word_idx)

            for context_idx in context_indices:
                c_neg = []
                for _ in range(k):
                    neg_word_idx = None
                    while True:
                        if optmim_random_choice:
                            neg_word_idx = np.random.choice(len(vocab_list), p=probabilities)
                        else:
                            neg_word = random.choice(vocab_list)
                            neg_word_idx = word_to_index.get(neg_word, None)
                            if neg_word_idx is None:
                                continue
                        neg_word = vocab_list[neg_word_idx]
                        if neg_word != word and neg_word not in word_except:
                            break
                    c_neg.append(neg_word_idx)
                training_data.append([target_idx, context_idx] + c_neg)

    return training_data

