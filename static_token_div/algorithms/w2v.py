""" This script is an embedding file generator
.module author:: Marius THORRE
"""

import static_token_div.tools.tools as tools


def embedding_generator(
    training_path: str,
    text_path: str,
    L: int,
    k: int,
    minc: int,
    word_except: list,

) -> None:
    """
    Embedding generator file
    :param training_path: embedding saving path file
    :param text_path: text to apply embeddings
    :param L: right and left context size, windows size is equal to 2*L + 1
    :param k: negative nb context for a positive context
    :param minc: minimal occurrence number
    :param word_except: word except, will not appear as embedding target
    :return: None
    """

    text = tools._get_text(text_path)
    vocab = tools.create_vocabulary2(text)
    print(f"\t Vocab size: {len(vocab)}")
    context = tools._create_context(text, vocab, L, k, word_except, minc)

    tools._generate_training_data(context, training_path)