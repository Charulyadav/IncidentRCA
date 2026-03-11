import json
from pathlib import Path
from data import PAD_IDX, prepare_Data
import torch
from torch.utils.data import DataLoader
from models import LSTMTextClassifier
from typing import Counter, Dict, List, Tuple
from utils import EarlyStopper
import yaml


def train_one_epoch( model: LSTMTextClassifier, loader: DataLoader, criterion: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model: LSTMTextClassifier, loader: DataLoader, criterion: torch.nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


def save_vocab(vocab: Dict[str, int], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)


def load_config(config_path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = prepare_Data(
        train_csv_path=config["train_csv"],
        val_csv_path=config["val_csv"],
        text_col=config["text_col"],
        label_col=config["label_col"]
    )
    train_loader, val_loader, num_classes = data.get_DataLoaders(config["batch_size"])
    print(f"Loaded data with {num_classes} classes. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print("vocab size:", len(data.vocab))

    model = LSTMTextClassifier(
        vocab_size=len(data.vocab),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=num_classes,
        num_layers=config["num_layers"],
        bidirectional=config["bidirectional"],
        pad_idx=PAD_IDX,
        dropout=config["dropout"],
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_lstm_classifier.pt"

    early_stopper = EarlyStopper(patience=config["patience"], min_delta=config["min_delta"])

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        should_stop = early_stopper(val_acc, model.state_dict())
        print(
            f"Epoch {epoch}/{config['epochs']} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if early_stopper.improved:
            print(f"Saving new best model with val_acc={val_acc:.4f}")
            torch.save(model.state_dict(), model_path)

        if should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"Best validation accuracy: {early_stopper.best_accuracy:.4f}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
