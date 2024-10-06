import numpy as np
from static_token_div.tools.tools import sigmoid


def read_vocab(vocab_path: str):
    vocab = []
    with open(vocab_path, "r") as f:
        for index in f.readlines():
            vocab.append(int(index.replace("\n", "")))
    return np.array(vocab)


def read_example(exemple_path: str):
    exemple = {}
    with open(exemple_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "").split(" ")
            line = [int(a) for a in line]
            exemple[line[0]] = list(line[1:])
    return exemple


def create_matrices(example: dict, vocab_size: int, embedding_size: int):
    W = np.ones([embedding_size, vocab_size])
    C = []
    for word in example.keys():
        line = example.get(word)
        C.append(np.array(line))
    return np.array(C), W


def loss(m: np.ndarray, cpos: np.ndarray, cneg: np.ndarray):
    print((-np.log(sigmoid(m * cpos))).shape)
    # return -(np.log(sigmoid(m *cpos) + np.sum(np.log(sigmoid(-m * cneg)))))
    return 0


def update(
        W: np.ndarray,
        C: np.ndarray,
        cpos_index: int,
        cneg_index: int,
        iteration: int = 1,
        learning_rate: float = 1e-3
):
    losses = []
    for _ in range(iteration):
        print(W.shape)
        for m_index in range(W.shape[0]):
            cpos = C[m_index, :cpos_index]
            cneg = C[m_index, cpos_index:]
            print(cpos.shape)
            print(cneg.shape)
         #   m = W[m_index].reshape(-1, 1)
          #  print(m.shape)

        '''zpos = (cpos @ m).reshape(-1, 1)
        cpos = cpos - learning_rate * (zpos - 1) @ m.T

        zneg = m @ (m.T @ cneg)
        cneg = cneg - learning_rate * (sigmoid(zneg) - 1)

        a = learning_rate * cpos @ (sigmoid(zpos) - 1)
        b = sigmoid(m.T @ cneg) @ cneg.T
        m = m - a + b.T
        losses.append(loss(m, cpos, cneg))'''

    return losses


if __name__ == "__main__":
    vocab_path = ""
    examples_path = ""
    vocab = read_vocab(vocab_path)
    example = read_example(examples_path)
    C, W = create_matrices(example=example, vocab_size=vocab.shape[0], embedding_size=100)
    losses = update(W=W, C=C, cpos_index=100, cneg_index=100)
    print(losses)
