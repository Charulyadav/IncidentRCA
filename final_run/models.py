from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor


class LSTMTextClassifier(nn.Module):
	"""Multilayer LSTM text classifier with learnable embeddings.

	Expected input:
	- input_ids: LongTensor of shape (batch_size, seq_len)
	- lengths: Optional LongTensor of shape (batch_size,) with true sequence lengths

	Output:
	- logits: FloatTensor of shape (batch_size, num_classes)
	"""

	def __init__(
		self,
		vocab_size: int,
		embedding_dim: int,
		hidden_dim: int,
		num_classes: int,
		num_layers: int = 2,
		bidirectional: bool = False,
		pad_idx: int = 0,
		dropout: float = 0.3,
	) -> None:
		super().__init__()

		self.embedding = nn.Embedding(
			num_embeddings=vocab_size,
			embedding_dim=embedding_dim,
			padding_idx=pad_idx,
		)
		self.embedding_dropout = nn.Dropout(dropout)

		# LSTM dropout is applied between stacked LSTM layers only.
		lstm_dropout = dropout if num_layers > 1 else 0.0
		self.lstm = nn.LSTM(
			input_size=embedding_dim,
			hidden_size=hidden_dim,
			num_layers=num_layers,
			batch_first=True,
			dropout=lstm_dropout,
			bidirectional=bidirectional,
		)

		lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
		self.classifier = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(lstm_output_dim, num_classes),
		)

	def forward(self, input_ids: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
		x = self.embedding(input_ids)
		x = self.embedding_dropout(x)

		if lengths is not None:
			packed = nn.utils.rnn.pack_padded_sequence(
				x,
				lengths.cpu(),
				batch_first=True,
				enforce_sorted=False,
			)
			_, (h_n, _) = self.lstm(packed)
		else:
			_, (h_n, _) = self.lstm(x)

		# h_n shape: (num_layers * num_directions, batch, hidden_dim)
		if self.lstm.bidirectional:
			# Concatenate final forward and backward states from the last layer.
			features = torch.cat((h_n[-2], h_n[-1]), dim=1)
		else:
			features = h_n[-1]

		logits = self.classifier(features)
		return logits


if __name__ == "__main__":
	# Quick sanity check
	batch_size = 4
	seq_len = 10
	model = LSTMTextClassifier(
		vocab_size=5000,
		embedding_dim=128,
		hidden_dim=256,
		num_classes=5,  # n classes
		num_layers=2,
		bidirectional=True,
		pad_idx=0,
		dropout=0.3,
	)

	input_ids = torch.randint(1, 4999, (batch_size, seq_len))
	lengths = torch.tensor([10, 9, 7, 5])
	logits = model(input_ids, lengths)
	print("Logits shape:", logits.shape)  # (batch_size, num_classes)
