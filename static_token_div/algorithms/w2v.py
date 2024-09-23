""" This script is an embedding file generator
.module author:: Marius THORRE
"""

import static_token_div.tools.tools as tools


def embedding_generator(
    save_path: str,
    text_path: str,
    L: int,
    k: int,
    eta: int,
    e: int,
    minc: int,
    word_except: list

) -> None:
    """
    Embedding generator file
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
    occurrences = tools.get_occurrences(text, vocab, minc)
    embeddings = tools.create_embeddings(text, vocab, occurrences, L, word_except)
    pos_context = tools.create_pos_context(embeddings, vocab)
    neg_context = tools.create_neg_context(pos_context, vocab, k)

    to_save = ""
    for main_word in pos_context.keys():
        for pos_word in pos_context.get(main_word):
            to_save += str(main_word) + " " + str(pos_word) + "\n"

    to_save += "-----separation------" + "\n"

    for main_word in neg_context.keys():
        for neg_word in neg_context.get(main_word):
            to_save += str(main_word) + " " + str(neg_word) + "\n"

    print(f"\tFile contains: {len(embeddings)} lines")
    with open(save_path, "w", encoding='utf-8') as f:
        f.write(to_save)