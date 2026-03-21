import json
import argparse
from pathlib import Path
from typing import Dict, List
import torch
import yaml
from models import LSTMTextClassifier

UNK_IDX = 1

def simple_tokenize(text: str):
    return str(text).lower().split()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_vocab(vocab_path: str) -> Dict[str, int]:
    """Load vocabulary mapping from JSON file."""
    with open(vocab_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_label_mappings(label_map_path: str) -> Dict[str, str]:
    """Load label mappings from JSON file."""
    label_path = Path(label_map_path)
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as file:
            label_map = json.load(file)
        return label_map.get("id_to_label", {})
    return {}


def encode_text(text: str, vocab: Dict[str, int]) -> List[int]:
    """
    Encode text to token IDs using vocabulary.
    
    Args:
        text: Input text to encode
        vocab: Vocabulary mapping from tokens to indices
        
    Returns:
        List of token IDs
    """
    tokens = simple_tokenize(text)
    if not tokens:
        return [UNK_IDX]
    return [vocab.get(tok, UNK_IDX) for tok in tokens]


def predict_single_text(
    text: str,
    model: LSTMTextClassifier,
    vocab: Dict[str, int],
    id_to_label: Dict[str, str],
    device: torch.device,
) -> tuple[str, float]:
    """
    Predict category for a single piece of text.
    
    Args:
        text: Input text to classify
        model: Trained LSTM classifier model
        vocab: Vocabulary mapping
        id_to_label: Mapping from class ID to label name
        device: Device to run model on (cuda or cpu)
        
    Returns:
        Tuple of (predicted_label, predicted_label_confidence)
    """
    # Encode text
    encoded = encode_text(text, vocab)
    
    # Convert to tensor and add batch dimension
    input_ids = torch.tensor([encoded], dtype=torch.long)  # Shape: (1, seq_len)
    lengths = torch.tensor([len(encoded)], dtype=torch.long)  # Shape: (1,)
    
    # Move to device
    input_ids = input_ids.to(device)
    lengths = lengths.to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, lengths)  # Shape: (1, num_classes)
        probabilities = torch.softmax(logits, dim=1)[0]  # Shape: (num_classes,)
        predicted_class_id = logits.argmax(dim=1).item()  # Scalar
    
    # Map to label
    predicted_label = id_to_label.get(str(predicted_class_id), str(predicted_class_id))
    predicted_confidence = probabilities[predicted_class_id].item()
    
    return predicted_label, predicted_confidence


def main(input_text: str) -> tuple[str, str, float]:
    """Return input text, predicted category, and confidence for that category."""
    # Setup paths
    config_path = "config.yaml"
    model_path = "best_lstm_classifier.pt"
    vocab_path = "vocab.json"
    label_map_path = "label_mappings.json"
    
    # Load configuration and data
    config = load_config(config_path)
    vocab = load_vocab(vocab_path)
    id_to_label = load_label_mappings(label_map_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    classifier_weight = checkpoint["classifier.1.weight"]
    num_classes = classifier_weight.size(0)
    
    model = LSTMTextClassifier(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=num_classes,
        num_layers=config["num_layers"],
        bidirectional=config["bidirectional"],
        pad_idx=0,
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint)

    predicted_label, predicted_confidence = predict_single_text(
        input_text,
        model,
        vocab,
        id_to_label,
        device,
    )
    return input_text, predicted_label, predicted_confidence


if __name__ == "__main__":
    # sample text1 = "Database connection timeout error during query execution"
    # sample text2 = "Server out of memory issue causing application crash"
    # sample text3 = "Firewall blocking incoming network traffic"
    parser = argparse.ArgumentParser(description="Classify a single text with the saved LSTM model")
    parser.add_argument("text", type=str, help="Input text to classify")
    cli_args = parser.parse_args()

    model_input_text, predicted_category, confidence = main(cli_args.text)
    print(f"Input Text: {model_input_text}")
    print(f"Predicted Category: {predicted_category}")
    print(f"Confidence: {confidence*100:.4f}%")
