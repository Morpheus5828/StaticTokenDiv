
import numpy as np


def get_text(text_path: str) -> list[str]:
    text_path = text_path.lower()
    with open(text_path, "r", encoding='utf-8') as vocab_file:
        text = vocab_file.read().split("\n")
    return text


def sigmoid(
        z: np.ndarray,
) -> float:
    return 1 / (1 + np.exp(-z))



'''def break_list_for_txt(embedding: list) -> str:
    result = ""
    for index, word in enumerate(embedding):
        if index == len(embedding) - 1:
            result += str(word)
        else:
            result += str(word) + " "
    return result'''



'''def generate_vocab_file(
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





def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def extract_embeddings_data(
        txt_path: str,
):
    df = pd.read_csv(txt_path, delimiter=' ')
    return df



def create_vocabulary2(
        text: list
) -> dict:
    word_counts = defaultdict(int)
    for sentence in text:
        for word in sentence.split():
            word_counts[word] += 1
        vocab = {word: (idx, count) for idx, (word, count) in enumerate(word_counts.items())}
    return vocab'''

