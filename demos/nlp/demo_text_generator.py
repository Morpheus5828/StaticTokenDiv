""" This script is a demo script generate text using NLP model
..moduleauthor::Marius THORRE
"""

import torch
import os, sys
module_path = os.path.abspath(os.path.join('../..'))

if module_path not in sys.path:
    sys.path.append(module_path)

import static_token_div.learning.text_generator as text_generator
from static_token_div.learning.models import NLP_SGNS, NLP_embedding
import static_token_div.tools.vocab_tools as vocab_tools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\t Using gpu: %s ' % torch.cuda.is_available())

corpus_path = os.path.join("../../resources/tlnl_tp1_data/alexandre_dumas/fusion.txt")

embedding_path = "fusion_emb_filename.txt"
vocab_NLP_SGNS = vocab_tools.Vocab(emb_filename=embedding_path)


def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = text.strip().split()
    return tokens


def create_vocab(tokens):
    vocab = {}
    for word in tokens:
        if word not in vocab.keys():
            vocab[word] = len(vocab)
    return vocab

corpus = read_corpus(corpus_path)
vocab_NLP_embedding = create_vocab(corpus)

NLP_SGNS_model = NLP_SGNS(k=5, vocab_size=vocab_NLP_SGNS.vocab_size + 1)
NLP_SGNS_model.load_state_dict(torch.load('NLP_SGNS_perplexity_fusion.pth', map_location=torch.device('cpu')))

NLP_embedding_model = NLP_embedding(k=5, vocab_size=len(vocab_NLP_embedding.keys()))
NLP_embedding_model.load_state_dict(torch.load('NLP_embedding_model_fusion.pth', map_location=torch.device('cpu')))


k = 5 # fixed
start_sentence = "le poisson dans l eau"
blacklist = ['<s>', '</s>', '<unk>', ",", ".", "...", ";"]

print(f"\t Generation using NLP_SGNS model:\n")
result = text_generator.run(
    start_word=start_sentence,
    k=k,
    model=NLP_SGNS_model,
    device=device,
    vocab=vocab_NLP_SGNS,
    blacklist=blacklist,
    n_words=4
)

for key in result.keys():
    print(key, result[key])

print(f"\t Generation using NLP_embedding model:\n")

result = text_generator.run_for_NLP_embedding(
    start_word=start_sentence,
    k=k,
    model=NLP_embedding_model,
    device=device,
    vocab=vocab_NLP_embedding,
    blacklist=blacklist,
    n_words=4
)

for key in result.keys():
    print(key, result[key])





