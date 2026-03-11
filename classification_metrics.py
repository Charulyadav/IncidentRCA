import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from data import PAD_IDX, TextClassificationDataset
from models import LSTMTextClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the saved LSTM classifier on a labeled test set")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model-path", type=str, default="outputs/best_lstm_classifier.pt")
    parser.add_argument("--vocab-path", type=str, default="outputs/vocab.json")
    parser.add_argument("--label-map-path", type=str, default="data/processed/label_mappings.json")
    parser.add_argument("--test-csv", type=str, default=None)
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_vocab(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_label_names(label_map_path: str, num_classes: int) -> List[str]:
    label_path = Path(label_map_path)
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as file:
            label_map = json.load(file)
        id_to_label = label_map.get("id_to_label", {})
        return [id_to_label.get(str(index), str(index)) for index in range(num_classes)]
    return [str(index) for index in range(num_classes)]


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    lengths = torch.tensor([sequence.size(0) for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(list(sequences), batch_first=True, padding_value=PAD_IDX)
    labels_tensor = torch.stack(labels)
    return padded, lengths, labels_tensor


def build_test_loader(test_csv: str, text_col: str, label_col: str, vocab: Dict[str, int], batch_size: int) -> DataLoader:
    test_df = pd.read_csv(test_csv)
    test_texts = test_df[text_col].astype(str).tolist()
    test_labels = test_df[label_col].astype(int).tolist()
    test_dataset = TextClassificationDataset(test_texts, test_labels, vocab)

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )


@torch.no_grad()
def evaluate_model(model: LSTMTextClassifier, loader: DataLoader, device: torch.device) -> Tuple[List[int], List[int]]:
    model.eval()
    all_predictions: List[int] = []
    all_labels: List[int] = []

    for input_ids, lengths, labels in loader:
        logits = model(input_ids.to(device), lengths.to(device))
        predictions = logits.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())

    return all_labels, all_predictions


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_csv = args.test_csv or config["val_csv"]
    vocab = load_vocab(args.vocab_path)
    checkpoint = torch.load(args.model_path, map_location=device)

    classifier_weight = checkpoint["classifier.1.weight"]
    num_classes = classifier_weight.size(0)

    model = LSTMTextClassifier(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=num_classes,
        num_layers=config["num_layers"],
        bidirectional=config["bidirectional"],
        pad_idx=PAD_IDX,
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint)

    test_loader = build_test_loader(
        test_csv=test_csv,
        text_col=config["text_col"],
        label_col=config["label_col"],
        vocab=vocab,
        batch_size=config["batch_size"],
    )
    label_names = load_label_names(args.label_map_path, num_classes)

    y_true, y_pred = evaluate_model(model, test_loader, device)

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print(f"Test file: {test_csv}")
    print(f"Model path: {args.model_path}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nPer-class report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=label_names,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()