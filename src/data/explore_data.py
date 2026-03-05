import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
OUTPUTS = PROJECT_ROOT / "outputs"

def load_data():
    """Load the raw incident dataset"""
    df = pd.read_csv(DATA_RAW / "incidents.csv")
    print(f"Dataset loaded: {len(df)} records")
    return df

def basic_info(df):
    """Print basic dataset information"""
    print("\nBASIC DATASET INFO")
    print("=" * 50)

    # Shape of dataset
    print(f"\nRows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Column names and types
    print(f"\nColumn Data Types:")
    print(df.dtypes)

    # Missing values
    print(f"\nMissing Values:")
    print(df.isnull().sum())


def analyze_categories(df):
    """Analyze the distribution of issue categories"""
    print("\nCATEGORY ANALYSIS")
    print("=" * 50)

    # Count each category
    category_counts = df['category'].value_counts()

    print(f"\nUnique categories: {len(category_counts)}")
    print(f"\nCategory Distribution:")
    print(category_counts)

    # Visualize distribution
    plt.figure(figsize=(10, 8))
    sns.barplot(x=category_counts.values, y=category_counts.index, palette="viridis")
    plt.xlabel("Number of Records")
    plt.ylabel("Category")
    plt.title("Distribution of Issue Categories")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "category_distribution.png", dpi=150)
    plt.show()
    print(f"\nPlot saved to: {OUTPUTS / 'category_distribution.png'}")


def analyze_priority(df):
    """Analyze priority distribution"""

    print("\nPRIORITY ANALYSIS")
    print("=" * 50)

    # Count each priority level
    priority_counts = df['priority'].value_counts()

    print(f"\nPriority Distribution:")
    print(priority_counts)

    # Visualize
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='priority', palette="rocket")
    plt.xlabel("Priority")
    plt.ylabel("Count")
    plt.title("Distribution of Priority Levels")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "priority_distribution.png", dpi=150)
    plt.show()


def analyze_text_fields(df):
    """Analyze text fields"""
    print("\nTEXT FIELD ANALYSIS")
    print("=" * 50)

    # Error_Description length analysis
    df['desc_word_count'] = df['Error_Description'].apply(lambda x: len(str(x).split()))
    df['desc_char_count'] = df['Error_Description'].apply(lambda x: len(str(x)))

    print("\nError_Description - Word Count:")
    print(df['desc_word_count'].describe())

    print("\nError_Description - Character Count:")
    print(df['desc_char_count'].describe())

    # Error_name analysis
    print(f"\nUnique Error Names: {df['Error_name'].nunique()}")
    print(f"\nTop 10 Error Names:")
    print(df['Error_name'].value_counts().head(10))


def check_short_descriptions(df):
    """Check for problematic short descriptions"""
    print("\nSHORT DESCRIPTION CHECK")
    print("=" * 50)

    # Find descriptions with less than 5 words
    short_desc = df[df['Error_Description'].apply(lambda x: len(str(x).split()) < 5)]

    print(f"\nDescriptions with < 5 words: {len(short_desc)}")

    if len(short_desc) > 0:
        print("\nSamples of short descriptions:")
        for idx, row in short_desc.head(10).iterrows():
            print(f"\n  Error_name: {row['Error_name']}")
            print(f"  Description: {row['Error_Description']}")
            print(f"  Category: {row['category']}")


def generate_summary(df):
    """Generate a summary report of data exploration"""

    print("\nDATA EXPLORATION SUMMARY")
    print("=" * 50)

    # Calculate stats
    category_counts = df['category'].value_counts()
    rare_categories = category_counts[category_counts < 20]

    short_desc_count = len(df[df['Error_Description'].apply(lambda x: len(str(x).split()) < 5)])

    priority_values = df['priority'].unique()

    print(f"""
    DATASET OVERVIEW:
    - Total records: {len(df)}
    - Total columns: {len(df.columns)}
    - Missing values: {df.isnull().sum().sum()}

    CATEGORY (Target Variable):
    - Unique categories: {len(category_counts)}
    - Largest category: {category_counts.index[0]} ({category_counts.iloc[0]} records)
    - Smallest category: {category_counts.index[-1]} ({category_counts.iloc[-1]} records)
    - Rare categories (<20 samples): {len(rare_categories)}

    PRIORITY (Input Feature):
    - Unique values: {len(priority_values)}
    - Values found: {list(priority_values)}
    - Issue: Inconsistent casing (e.g., 'High' vs 'high')

    TEXT FIELDS:
    - Unique Error_names: {df['Error_name'].nunique()}
    - Avg description length: {df['Error_Description'].apply(lambda x: len(str(x).split())).mean():.1f} words
    - Short descriptions (<5 words): {short_desc_count}

    DATA QUALITY ISSUES TO ADDRESS:
    1. Standardize priority casing (High, Medium, Low, Critical)
    2. Handle rare categories (merge into 'Other' or remove)
    3. Decide on short descriptions (keep or remove)
    """)

if __name__ == "__main__":
    df = load_data()

    # print(df.head())
    # print(f"Columns: {list(df.columns)}")
    # basic_info(df)
    # analyze_categories(df)
    # analyze_priority(df)
    # analyze_text_fields(df)
    # check_short_descriptions(df)
    generate_summary(df)
    # Check for acronyms in Error_name
    acronyms = df[df['Error_name'].str.contains(r'\b[A-Z]{2,}\b', regex=True)]
    print(f"Records with acronyms: {len(acronyms)}")
    print(acronyms['Error_name'].head(10))
