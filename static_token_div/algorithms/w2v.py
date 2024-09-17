""" This script is an embedding file generator
.module author:: Marius THORRE
"""

from static_token_div.tools.tools import get_embedding_sentence, break_list_for_txt


def embedding_generator(
    save_path: str,
    text_path: str,
    L: int,
    minc: int,
    word_except: list = ["<s>", "</s>"]

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
    text_embedding = get_embedding_sentence(
        text_path=text_path,
        L=L,
        minc=minc,
        word_except=word_except
    )
    embedding_to_save = ""
    for embedding in text_embedding:
        embedding_to_save += break_list_for_txt(embedding) + "\n"

    print(f"\tFile contains: {len(text_embedding)} lines")
    with open(save_path, "w", encoding='utf-8') as f:
        f.write(embedding_to_save)








