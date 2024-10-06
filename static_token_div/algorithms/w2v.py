""" This script is an embedding file generator
.module author:: Marius THORRE, Thomas CELESCHI
"""

import static_token_div.tools.tools as tools


def embedding_generator(
    vocab_path: str,
    embeddings_path: str,
    text_path: str,
    L: int,
    k: int,
    minc: int,
    word_except: list,
    max_context: int

) -> None:
    """
    Embedding generator file
    :param max_context:
    :param save_path2:
    :param save_path: embedding saving path file
    :param text_path: text to apply embeddings
    :param L: right and left context size, windows size is equal to 2*L + 1
    :param k: negative nb context for a positive context
    :param eta: learning rate
    :param e: nb iterations
    :param minc: minimal occurrence number
    :param word_except: word except, will not appear as embedding target
    :return: None
    """

    text = tools.get_text(text_path)
    vocab = tools.create_vocabulary(text)
    print(f"\t Vocab size: {len(vocab)}")
    occurrences = tools.get_occurrences(text, vocab, minc)
    embeddings = tools.create_embeddings(text, vocab, occurrences, L, word_except)
    pos_context = tools.create_pos_context(embeddings, vocab, occurrences, L, word_except)
    neg_context = tools.create_neg_context(pos_context, vocab, k, occurrences, word_except)

    tools.generate_vocab_file(vocab, occurrences, word_except, vocab_path)

    tools.generate_embeddings_file(pos_context, neg_context, embeddings, embeddings_path)