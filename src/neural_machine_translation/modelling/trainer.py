# sys libs
import os
import sys
import random
import warnings

warnings.filterwarnings("ignore")

# data manupulation libs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# string manupulation libs
import re
import string
from string import digits
import spacy

import math
import time
import pandas as pd

# torch libs
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.nn.parallel import DataParallel

from torch import Tensor
from torch.nn import Transformer


from timeit import default_timer as timer

from neural_machine_translation.preprocess.dataloader import (
    master_loader,
    last_loader,
    val_loaders,
)


from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge


src_lang = "bn"
trg_lang = "hi"

train = pd.read_csv(f"/content/train_df_{src_lang}_{trg_lang}.csv")
val = pd.read_csv(f"/content/test_df_{src_lang}_{trg_lang}.csv")

train.head()


batch_size = 128
train_iterator, val_iterator, src_len, trg_len, trg_vocab = master_loader(
    train, val, 2, src_lang, trg_lang, batch_size
)
print(src_len, trg_len)


SRC_VOCAB_SIZE = src_len
TRG_VOCAB_SIZE = trg_len
D_MODEL = 256

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


#   Get these from the .env file
EMB_SIZE = 128
NHEAD = 8
FFN_HID_DIM = 128
BATCH_SIZE = batch_size
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
PAD_IDX = 0

transformer = Seq2SeqTransformer(
    NUM_ENCODER_LAYERS,
    NUM_DECODER_LAYERS,
    EMB_SIZE,
    NHEAD,
    SRC_VOCAB_SIZE,
    TRG_VOCAB_SIZE,
    FFN_HID_DIM,
)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0003, betas=(0.9, 0.98), eps=1e-9
)


def train_epoch(model, optimizer, train_iterator):
    model.train()
    losses = 0

    train_dataloader = train_iterator

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))


def evaluate(model, val_iterator):
    model.eval()
    losses = 0

    val_dataloader = val_iterator

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))




NUM_EPOCHS = 100

for epoch in range(1, NUM_EPOCHS + 1):
    print(epoch)
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer, train_iterator)
    end_time = timer()
    print(
        (
            f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "
            f"Epoch time = {(end_time - start_time):.3f}s"
        )
    )

print("Done")
# Save the model after all epochs are done
torch.save(transformer.state_dict(), f"model_{src_lang}_{trg_lang}.pth")


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(
            DEVICE
        )
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == 2:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src):
    model.eval()
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=1
    ).flatten()
    return tgt_tokens



def create_dataframe(text):
    data = {"id": [10], "src": [text], "trg": [text]}
    df = pd.DataFrame(data)
    return df


val_df = pd.read_csv(f"/content/test_df_{src_lang}_{trg_lang}.csv")
fake_dataset, fake_loader = last_loader(train, 2, src_lang, trg_lang, 1)

results_df = []

start = time.time()
for index, row in tqdm(val_df.iterrows()):
    text = row["src"]
    _id = row["id"]

    result_df = create_dataframe(text)

    val_iterator = val_loaders(fake_dataset, fake_loader, result_df, 1)

    first_batch = next(iter(val_iterator))
    a = first_batch[0]
    b = translate(transformer, a)

    # Convert tensor of indexes to a list of words using the vocabulary, excluding indices 0, 1, and 2
    words = [trg_vocab[idx.item()] for idx in b if idx.item() not in {0, 1, 2}]

    # Join the list of words into a string
    result_string = " ".join(words)

    results_df.append([_id, f'"{result_string}"'])

print("Done")

# Sort the DataFrame by 'id' in ascending order
results = pd.DataFrame(results_df, columns=["ID", "Translation"])

# Save the DataFrame to a CSV file
results.to_csv(f"output_{src_lang}_{trg_lang}.csv", index=False)

print("Done")

print(time.time() - start)


# BLEU score calculation function
def calculate_bleu_score(reference, translation):
    return corpus_bleu(
        [[ref.split()] for ref in reference], [trans.split() for trans in translation]
    )


# ROUGE score calculation function
def calculate_rouge_score(reference, translation):
    rouge = Rouge()
    scores = rouge.get_scores(translation, reference)
    rouge_scores = [score["rouge-l"]["f"] for score in scores]
    return sum(rouge_scores) / len(rouge_scores)


calculate_bleu_score(val["trg"], results["Translation"])

calculate_rouge_score(val["trg"], results["Translation"])
