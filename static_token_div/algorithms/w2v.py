""" This script is an embedding file generator
.module author:: Marius THORRE, Thomas CELESCHI
"""

import static_token_div.tools.tools as tools


def embedding_generator(
    save_path1: str,
    save_path2: str,
    text_path: str,
    L: int,
    k: int,
    minc: int,
    word_except: list

) -> None:
    """
    Embedding generator file
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
    pos_context = tools.create_pos_context(embeddings, vocab)
    neg_context = tools.create_neg_context(pos_context, vocab, k)

    to_save = ""
    for main_word in vocab.keys():
        if main_word in occurrences and main_word not in word_except:
            to_save += str(vocab.get(main_word)) + "\n"

    with open(save_path1, "w", encoding='utf-8') as f:
        f.write(to_save)

    to_save2 = ""
    for main_word in vocab.keys():
        if main_word in occurrences and main_word not in word_except:
            word = vocab.get(main_word)
            to_save2 += str(word) + " "
            pos_list = list(pos_context.get(word))
            if len(pos_list) >= 100:
                for i in range(100):
                    to_save2 += str(pos_list[i]) + " "
            else:
                for i in range(len(pos_list)):
                    to_save2 += str(pos_list[i]) + " "
                for i in range(100 - len(pos_list)):
                    to_save2 += str(0) + " "
            neg_list = list(neg_context.get(word))
            if len(pos_list) > 100:
                for i in range(k):
                    for j in range(100):
                        to_save2 += str(neg_list[i*j]) + " "
            else:
                for i in range(k):
                    for j in range(len(pos_list)):
                        to_save2 += str(neg_list[i*j]) + " "
                    for j in range(100 - len(pos_list)):
                        to_save2 += str(0) + " "
            to_save2 += "\n"

    # for main_word in pos_context.keys():
    #     for pos_word in pos_context.get(main_word):
    #         to_save += str(main_word) + " " + str(pos_word) + "\n"
    #
    # for main_word in neg_context.keys():
    #     for neg_word in neg_context.get(main_word):
    #         to_save += str(main_word) + " " + str(neg_word) + "\n"

    print(f"\tFile contains: {len(embeddings)} lines")
    with open(save_path2, "w", encoding='utf-8') as f2:
        f2.write(to_save2)