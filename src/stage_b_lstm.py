# BiLSTM sentiment classifier (deep-learning Stage B baseline).
# GloVe-initialised embedding -> bidirectional LSTM -> MLP head.
# Loss is BCEWithLogits (binary pos/neg). Early stops on val loss.
import time

import gensim.downloader as gensim_downloader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .preprocess import tokenize_corpus


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


def build_vocab(tokenised, min_count=2, max_size=40000):
    # Frequency-ranked vocab with reserved PAD=0 and UNK=1.
    counts = {}
    for tokens in tokenised:
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
    items = [(tok, c) for tok, c in counts.items() if c >= min_count]
    items.sort(key=lambda x: -x[1])
    items = items[:max_size]
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for tok, _ in items:
        vocab[tok] = len(vocab)
    return vocab


def numericalise(tokens, vocab, max_len):
    ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in tokens][:max_len]
    return ids


def init_embedding_from_glove(vocab, glove_name='glove-wiki-gigaword-100'):
    # Initialise embedding rows from GloVe; OOV rows stay random Gaussian.
    wv = gensim_downloader.load(glove_name)
    dim = wv.vector_size
    matrix = np.random.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    matrix[vocab[PAD_TOKEN]] = 0.0
    hits = 0
    for token, idx in vocab.items():
        if token in wv.key_to_index:
            matrix[idx] = wv[token]
            hits += 1
    return matrix, hits, dim


class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_pad(batch, pad_idx=0):
    # Pads each batch to its own longest sequence; lengths are kept for pack_padded.
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = lengths.max().item()
    padded = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded, lengths, labels


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_layers=1,
                 dropout=0.5, pad_idx=0, pretrained_embeddings=None,
                 freeze_embeddings=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = not freeze_embeddings
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))
        # Packing skips padded steps so the LSTM only sees real tokens.
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        # Concatenate the final forward and backward hidden states.
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        out = self.dropout(F.relu(self.fc1(combined)))
        return self.fc2(out).squeeze(1)


class BiLSTMSentiment:
    def __init__(self, embed_dim=100, hidden_dim=128, dropout=0.5,
                 max_len=400, batch_size=32, epochs=15, lr=1e-3,
                 use_glove=True, freeze_embeddings=False, random_state=42,
                 device=None, patience=2):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.use_glove = use_glove
        self.freeze_embeddings = freeze_embeddings
        self.random_state = random_state
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.patience = patience
        self.vocab = None
        self.model = None
        self.history = []
        self.best_epoch = None

    def _set_seed(self):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _to_sequences(self, texts):
        tokenised = tokenize_corpus(texts)
        return [numericalise(t, self.vocab, self.max_len) for t in tokenised]

    def fit(self, texts, labels, val_texts=None, val_labels=None, verbose=True):
        self._set_seed()
        tokenised_train = tokenize_corpus(texts)
        self.vocab = build_vocab(tokenised_train)
        train_seqs = [numericalise(t, self.vocab, self.max_len) for t in tokenised_train]

        pretrained = None
        glove_hits = 0
        if self.use_glove:
            pretrained, glove_hits, glove_dim = init_embedding_from_glove(self.vocab)
            self.embed_dim = glove_dim

        self.model = BiLSTMClassifier(
            vocab_size=len(self.vocab),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            pad_idx=self.vocab[PAD_TOKEN],
            pretrained_embeddings=pretrained,
            freeze_embeddings=self.freeze_embeddings,
        ).to(self.device)

        if verbose:
            params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f'  vocab size: {len(self.vocab)}, glove hits: {glove_hits}/{len(self.vocab)}, trainable params: {params:,}')

        train_ds = TextDataset(train_seqs, labels.tolist() if hasattr(labels, 'tolist') else list(labels))
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=collate_pad
        )

        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_seqs = self._to_sequences(val_texts)
            val_ds = TextDataset(val_seqs, val_labels.tolist() if hasattr(val_labels, 'tolist') else list(val_labels))
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_pad
            )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        best_state = None
        epochs_since_improve = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            t0 = time.perf_counter()
            losses, correct, total = [], 0, 0
            for x, lengths, y in train_loader:
                x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x, lengths)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                losses.append(loss.item())
                preds = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == y.long()).sum().item()
                total += y.size(0)
            train_loss = np.mean(losses)
            train_acc = correct / total

            val_loss, val_acc = None, None
            if val_loader is not None:
                val_loss, val_acc = self._eval_loader(val_loader, criterion)
            elapsed = time.perf_counter() - t0
            self.history.append({
                'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                'val_loss': val_loss, 'val_acc': val_acc, 'time': elapsed,
            })

            improved = ''
            if val_loss is not None:
                # Track best weights by val loss for early stopping.
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    self.best_epoch = epoch
                    epochs_since_improve = 0
                    improved = '  *'
                else:
                    epochs_since_improve += 1

            if verbose:
                if val_acc is not None:
                    print(f'  epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}  '
                          f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({elapsed:.1f}s){improved}')
                else:
                    print(f'  epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}  ({elapsed:.1f}s)')

            if val_loader is not None and epochs_since_improve >= self.patience:
                if verbose:
                    print(f'  early stop: no val_loss improvement for {self.patience} epochs (best at epoch {self.best_epoch})')
                break

        if best_state is not None:
            # Restore the best-val checkpoint before returning.
            self.model.load_state_dict(best_state)

        return self

    def _eval_loader(self, loader, criterion):
        self.model.eval()
        losses, correct, total = [], 0, 0
        with torch.no_grad():
            for x, lengths, y in loader:
                x, lengths, y = x.to(self.device), lengths.to(self.device), y.to(self.device)
                logits = self.model(x, lengths)
                loss = criterion(logits, y)
                losses.append(loss.item())
                preds = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == y.long()).sum().item()
                total += y.size(0)
        return float(np.mean(losses)), correct / total

    def predict_proba(self, texts):
        seqs = self._to_sequences(texts)
        ds = TextDataset(seqs, [0] * len(seqs))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_pad)
        probs = []
        self.model.eval()
        with torch.no_grad():
            for x, lengths, _ in loader:
                x, lengths = x.to(self.device), lengths.to(self.device)
                logits = self.model(x, lengths)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs)

    def predict(self, texts):
        return (self.predict_proba(texts) >= 0.5).astype(int)
