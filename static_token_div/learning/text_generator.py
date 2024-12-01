""" This script contains code to generate text using models
..moduleauthor::Marius THORRE
"""

import torch
import torch.nn.functional as F


def _generate(
        model,
        vocab,
        embedding,
        start_word: list,
        theta: float = 0.2,
        device='cpu',
        blacklist=[]
):
    model.eval()
    with torch.no_grad():
        start_indices = [vocab.dico_voca.get(word, vocab.dico_voca.get('<unk>')) for word in start_word]
        embeddings = torch.cat([torch.tensor(embedding[idx], dtype=torch.float32) for idx in start_indices]).unsqueeze(0).to(device)
        output = model(embeddings)

        if blacklist:
            blacklist_indices = [vocab.dico_voca.get(word) for word in blacklist if word in vocab.dico_voca]
            output[:, blacklist_indices] = -float('inf')

        scaled_logits = output / theta

        probabilities = F.softmax(scaled_logits, dim=1)

        next_index = torch.multinomial(probabilities, 1).item()

        index_to_word = {idx: word for word, idx in vocab.dico_voca.items()}
        next_word = index_to_word.get(next_index, '<unk>')

        return next_word


def run(
    start_word: str,
    k: float,
    model,
    vocab,
    device,
    n_words,
    blacklist: list = ['<s>', '<unk>', ",", ".", "...", ";"],
    theta: list = [0.1, 0.5, 1, 1.5, 2]
) -> dict:
    result = {}
    for t in theta:
        words = start_word.split(" ")
        for i in range(n_words):
            current_words = words[-k:]
            next_word = _generate(
                model,
                vocab,
                embedding=vocab.matrice,
                start_word=current_words,
                theta=t,
                device=device,
                blacklist=blacklist
            )
            words.append(next_word)
        result[t] = words
    return result


def _generate_for_NLP_embedding(
    vocab_NLP_embedding,
    model,
    input_words,
    blacklist,
    n,
    theta,
    device

):
    inverse_vocab = {idx: word for word, idx in vocab_NLP_embedding.items()}
    model.eval()
    generated = []


    input_indices = [vocab_NLP_embedding[word] for word in input_words]

    for _ in range(n):
        input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            logits = output / theta

            if blacklist:
                blacklist_indices = [vocab_NLP_embedding.get(word, -1) for word in blacklist]
                blacklist_indices = [idx for idx in blacklist_indices if idx != -1]
                logits[:, blacklist_indices] = -float('inf')

            probabilities = F.softmax(logits, dim=1)
            next_index = torch.multinomial(probabilities, 1).item()

        next_word = inverse_vocab.get(next_index, "<UNK>")
        generated.append(next_word)
        input_indices = input_indices[1:] + [next_index]

    return generated


def run_for_NLP_embedding(
    start_word: str,
    k: float,
    model,
    vocab,
    device,
    n_words,
    blacklist: list = ['<s>', '<unk>', ",", ".", "...", ";"],
    theta: list = [0.1, 0.5, 1, 1.5, 2]
):
    result = {}
    for t in theta:
        words = start_word.split(" ")
        for i in range(n_words):
            current_words = words[-k:]

            next_word = _generate_for_NLP_embedding(
                model=model,
                vocab_NLP_embedding=vocab,
                input_words=current_words,
                n=1,
                device=device,
                theta=t,
                blacklist=blacklist
            )[0]
            words.append(next_word)

        result[t] = words
    return result






