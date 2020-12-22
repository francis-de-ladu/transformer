import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Transformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, n_enc, n_heads, dropout, n_classes):
        super(RteTransformer, self).__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.embed_layer = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model)
        self.tfm_encoders = [TransformerEncoder(max_seq_len, d_model, n_heads, dropout).to(device)
                             for _ in range(n_enc)]
        self.dropout = nn.Dropout(dropout)
        """
        self.fc = nn.Sequential(
            nn.Linear(max_seq_len * d_model, max_seq_len * d_model),
            nn.ReLU(),
            nn.Linear(max_seq_len * d_model, n_classes),
        )
        """
        self.fc = nn.Linear(max_seq_len * d_model, n_classes)

    def forward(self, inputs, seqs_len):
        embeddings = self.embed_layer(inputs)
        pos_encodings = self.pos_encoding(embeddings, seqs_len)
        out = self.dropout(embeddings + pos_encodings)
        for encoder in self.tfm_encoders:
            out = encoder(out)

        out = self.fc(out.view(-1, self.max_seq_len * self.d_model))
        return F.softmax(out, dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.d_model = d_model

        def pos_encoding_helper(pos, i, d_model):
            pos_encoding = pos / 10000 ** (2 * i / d_model)
            return pos_encoding

        # compute positional encoding values before application of sin and cos
        self.pos_enc = pos_encoding_helper(np.arange(max_len)[:, np.newaxis],
                                           np.arange(d_model)[np.newaxis, :],
                                           d_model)

        # apply sin to even indices and cos to odd indices of the positional encodings
        self.pos_enc[:, 0::2] = np.sin(self.pos_enc[:, 0::2])
        self.pos_enc[:, 1::2] = np.cos(self.pos_enc[:, 1::2])
        self.pos_enc = torch.Tensor(self.pos_enc).to(device)

    def forward(self, inputs, seqs_len):
        out = inputs * np.sqrt(self.d_model)
        out = out + self.pos_enc
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, max_len, d_model, n_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model

        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.norm_attn = nn.LayerNorm([max_len, d_model])
        self.linear_out = nn.Linear(d_model, d_model)
        self.norm_out = nn.LayerNorm([max_len, d_model])
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # attention layer
        attn_out = self.dropout(self.self_attn(inputs))
        attn_out = self.norm_attn(attn_out + inputs)
        # feed-forward layer
        out = self.dropout(self.linear_out(attn_out))
        out = self.norm_out(out + attn_out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.attn_heads = [SelfAttentionHead(d_model, n_heads).to(device)
                           for _ in range(n_heads)]
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, inputs):
        out = torch.cat([head(inputs) for head in self.attn_heads], dim=-1)
        out = self.linear_out(out)
        return out


class SelfAttentionHead(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SelfAttentionHead, self).__init__()
        self.normalize_mult = np.sqrt(d_model // n_heads)
        self.query_linear = nn.Linear(d_model, d_model // n_heads)
        self.key_linear = nn.Linear(d_model, d_model // n_heads)
        self.value_linear = nn.Linear(d_model, d_model // n_heads)

    def forward(self, inputs):
        queries = self.query_linear(inputs)
        keys = self.key_linear(inputs)
        values = self.value_linear(inputs)

        scores = torch.bmm(queries, keys.transpose(1, 2))
        attn = F.softmax(scores / self.normalize_mult, dim=-1)
        weighted = torch.bmm(attn, values)
        return weighted
