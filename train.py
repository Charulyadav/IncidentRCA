import argparse
import json
from pathlib import Path
from data import PAD_IDX, prepare_Data
import torch
from torch.utils.data import DataLoader
from models import LSTMTextClassifier
from typing import Counter, Dict, List, Tuple
from utils import EarlyStopper


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multilayer LSTM text classifier")
    parser.add_argument("--train-csv", type=str, default="data/processed/train_minimal.csv")
    parser.add_argument("--val-csv", type=str, default="data/processed/test_minimal.csv")
    parser.add_argument("--text-col", type=str, default="clean_col")
    parser.add_argument("--label-col", type=str, default="category_encoded")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--output-dir", type=str, default="outputs/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = prepare_Data(
        train_csv_path=args.train_csv,
        val_csv_path=args.val_csv,
        text_col=args.text_col,
        label_col=args.label_col
    )
    train_loader, val_loader, num_classes = data.get_DataLoaders(args.batch_size)
    print(f"Loaded data with {num_classes} classes. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print("vocab size:", len(data.vocab))
    save_vocab(data.vocab, args.output_dir + "vocab.json")
    exit(0)

    model = LSTMTextClassifier(
        vocab_size=len(data.vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        pad_idx=PAD_IDX,
        dropout=args.dropout,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_lstm_classifier.pt"
    vocab_path = output_dir / "vocab.json"


    early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        should_stop = early_stopper(val_acc, model.state_dict())
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if early_stopper.improved:
            print(f"Saving new best model with val_acc={val_acc:.4f}")
            torch.save(model.state_dict(), model_path)

        if should_stop:
            print(f"Early stopping triggered at epoch {epoch:02d}")
            break

    save_vocab(data.vocab, vocab_path)
    print(f"Best validation accuracy: {early_stopper.best_accuracy:.4f}")
    print(f"Saved model to: {model_path}")
    print(f"Saved vocab to: {vocab_path}")


if __name__ == "__main__":
    main()
