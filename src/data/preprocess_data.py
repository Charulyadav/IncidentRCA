import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

def load_data():
    """Load the raw incident dataset"""
    df = pd.read_csv(DATA_RAW / "incidents.csv")
    print(f"Loaded {len(df)} records")
    return df

def standardize_priority(df):
    """Standardize priority values to title case"""
    print("\nStandardizing priority..")

    # Show before
    print(f"Before: {df['priority'].unique()}")

    # Convert to title case (high -> High, MEDIUM -> Medium)
    df['priority'] = df['priority'].str.strip().str.title()

    # Show after
    print(f"After: {df['priority'].unique()}")
    return df


def merge_rare_categories(df, threshold=20):
    """Merge categories with fewer than threshold samples into 'Other'"""
    print(f"\nMerging rare categories (threshold: {threshold})...")

    # Count each category
    category_counts = df['category'].value_counts()

    # Find rare categories
    rare_categories = category_counts[category_counts < threshold].index.tolist()

    print(f"Rare categories found: {len(rare_categories)}")
    print(f"Categories to merge: {rare_categories}")

    # Replace rare categories with 'Other'
    df['category'] = df['category'].replace(rare_categories, 'Other')

    # Show new distribution
    print(f"\nNew category distribution:")
    print(df['category'].value_counts())

    return df


def clean_text(text):
    """Clean a single text string"""
    if pd.isna(text):
        return ""

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove extra whitespace (multiple spaces -> single space)
    text = ' '.join(text.split())

    return text


def preprocess_text(df):
    """Apply text cleaning to text columns"""
    print("\nCleaning text fields...")

    # Clean individual columns
    df['Error_name_clean'] = df['Error_name'].apply(clean_text)
    df['Error_Description_clean'] = df['Error_Description'].apply(clean_text)

    # Create combined text field
    df['combined_text'] = df['Error_name_clean'] + " " + df['Error_Description_clean']

    # Show sample
    print("\nSample cleaned text:")
    print(f"Original Error_name: {df['Error_name'].iloc[0]}")
    print(f"Cleaned Error_name: {df['Error_name_clean'].iloc[0]}")
    print(f"Combined text: {df['combined_text'].iloc[0][:100]}...")

    return df


def encode_labels(df):
    """Convert category names to numeric labels"""
    print("\nEncoding category labels...")

    # Get unique categories (sorted for consistency)
    categories = sorted(df['category'].unique())

    # Create mappings
    label_to_id = {label: idx for idx, label in enumerate(categories)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    # Apply encoding
    df['category_encoded'] = df['category'].map(label_to_id)

    # Show mapping
    print(f"\nLabel mapping ({len(categories)} categories):")
    for label, idx in label_to_id.items():
        count = len(df[df['category'] == label])
        print(f"  {idx}: {label} ({count} records)")

    return df, label_to_id, id_to_label


def split_data(df, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    print(f"\nSplitting data (test_size={test_size})...")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['category_encoded'],
        random_state=random_state
    )

    print(f"Training set: {len(train_df)} records")
    print(f"Test set: {len(test_df)} records")

    # Verify stratification
    print("\nCategory distribution preserved:")
    print(f"  Train - Class 1 (Database Issue): {len(train_df[train_df['category_encoded'] == 1])} records")
    print(f"  Test  - Class 1 (Database Issue): {len(test_df[test_df['category_encoded'] == 1])} records")

    return train_df, test_df


def save_data(train_df, test_df, label_to_id, id_to_label):
    """Save processed data to files"""
    print("\nSaving processed data...")

    # Create output directory if it doesn't exist
    DATA_PROCESSED.mkdir(exist_ok=True)

    # Select columns to save
    columns_to_save = [
        'ticket_id',
        'Error_name',
        'Error_Description',
        'priority',
        'category',
        'Error_name_clean',
        'Error_Description_clean',
        'combined_text',
        'category_encoded'
    ]

    # Save train and test sets
    train_df[columns_to_save].to_csv(DATA_PROCESSED / "train.csv", index=False)
    test_df[columns_to_save].to_csv(DATA_PROCESSED / "test.csv", index=False)

    # Save label mappings
    mappings = {
        'label_to_id': label_to_id,
        'id_to_label': {str(k): v for k, v in id_to_label.items()}
    }

    with open(DATA_PROCESSED / "label_mappings.json", "w") as f:
        json.dump(mappings, f, indent=2)

    print(f"\nFiles saved to: {DATA_PROCESSED}")
    print(f"  - train.csv ({len(train_df)} records)")
    print(f"  - test.csv ({len(test_df)} records)")
    print(f"  - label_mappings.json ({len(label_to_id)} categories)")


def verify_output():
    """Verify the saved files"""
    print("\nVerification")
    print("=" * 50)

    # Load and check train.csv
    train = pd.read_csv(DATA_PROCESSED / "train.csv")
    print(f"\ntrain.csv:")
    print(f"  Rows: {len(train)}")
    print(f"  Columns: {list(train.columns)}")

    # Load and check test.csv
    test = pd.read_csv(DATA_PROCESSED / "test.csv")
    print(f"\ntest.csv:")
    print(f"  Rows: {len(test)}")

    # Load and check mappings
    with open(DATA_PROCESSED / "label_mappings.json", "r") as f:
        mappings = json.load(f)
    print(f"\nlabel_mappings.json:")
    print(f"  Categories: {len(mappings['label_to_id'])}")

    print("\nAll files verified successfully!")

if __name__ == "__main__":ere
    df = load_data()
    df = standardize_priority(df)
    df = merge_rare_categories(df)
    df = preprocess_text(df)
    df, label_to_id, id_to_label = encode_labels(df)
    train_df, test_df = split_data(df)
    save_data(train_df, test_df, label_to_id, id_to_label)
    verify_output()
