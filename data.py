import pandas as pd
from torch import Tensor
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Counter, Dict, List, Tuple
from torch.nn.utils.rnn import pad_sequence
from utils import simple_tokenize

def load_csv_data(train_path, test_path):
    """
    Load training and test data from CSV files.

    Parameters:
    train_path (str): The path to the training CSV file.
    test_path (str): The path to the test CSV file.

    Returns:
    tuple: A tuple containing the training and test DataFrames, and the number of classes.
    """
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        num_classes = train_data['category_encoded'].nunique()
        return train_data, test_data, num_classes
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None, None, None

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int]) -> None:
        self.labels = labels
        self.encoded = [self.encode_text(t, vocab) for t in texts]

    def __len__(self) -> int:
        return len(self.labels)
    
    @staticmethod
    def encode_text(text: str, vocab: Dict[str, int]) -> List[int]:
        tokens = simple_tokenize(text)
        if not tokens:
            return [UNK_IDX]
        return [vocab.get(tok, UNK_IDX) for tok in tokens]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        token_ids = torch.tensor(self.encoded[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return token_ids, label


class prepare_Data:
    def __init__(self, train_csv_path, val_csv_path, text_col, label_col):
        self.train_csv = train_csv_path
        self.val_csv = val_csv_path
        self.text_col = text_col
        self.label_col = label_col
    
    @staticmethod
    def collate_batch(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
        sequences, labels = zip(*batch)
        lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
        padded = pad_sequence(list(sequences), batch_first=True, padding_value=PAD_IDX)
        labels_tensor = torch.stack(labels)
        return padded, lengths, labels_tensor

    @staticmethod
    def build_vocab(texts: List[str], min_freq: int = 1) -> Dict[str, int]:
        counter = Counter()
        for text in texts:
            counter.update(simple_tokenize(text))

        vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        for token, freq in counter.items():
            if freq >= min_freq and token not in vocab:
                vocab[token] = len(vocab)
        return vocab

    def get_DataLoaders(self, batch_size):
        train_df = pd.read_csv(self.train_csv)
        val_df = pd.read_csv(self.val_csv)
        num_classes = train_df[self.label_col].nunique()

        train_texts = train_df[self.text_col].astype(str).tolist()
        val_texts = val_df[self.text_col].astype(str).tolist()

        train_labels = train_df[self.label_col].astype(int).tolist()
        val_labels = val_df[self.label_col].astype(int).tolist()

        self.vocab = self.build_vocab(train_texts)
        train_ds = TextClassificationDataset(train_texts, train_labels, self.vocab)
        val_ds = TextClassificationDataset(val_texts, val_labels, self.vocab)

        train_loader = DataLoader(
                        train_ds,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=self.collate_batch,
                    )
        val_loader = DataLoader(
                        val_ds,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=self.collate_batch,
                    )

        return train_loader, val_loader, num_classes