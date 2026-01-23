# Standard library
import sys
import subprocess
import pkgutil
import csv
import os

# ---------------------------
# 1) INSTALL / VERIFY (Colab)
# ---------------------------

def install_and_verify_packages():
    """Run install commands (intended for Colab). If running outside Colab,
    this will attempt to pip-install packages using subprocess."""

    packages = [
        "transformers",
        "datasets",
        "sentence-transformers",
        "fastai[transformers]",
        "torch",
        "torchvision",
        "torchaudio",
        "kaggle",
        "tokenizers",
        "ipython",
    ]

    # Attempt to install packages via pip
    print("Installing packages (this may take several minutes)...")
    try:
        # Install all at once to resolve dependency conflicts (e.g. fastai requiring specific torch versions)
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade"] + packages, check=False)
    except Exception as e:
        print(f"Warning: pip install failed: {e}")

    # Verify installs (short)
    print("\nVerification (installed?):")
    for pkg in ("transformers","datasets","sentence_transformers","fastai","kaggle","torch", "IPython"):
        print(pkg, "installed:", pkgutil.find_loader(pkg) is not None)


# ---------------------------
# 2) OPTIONAL: KAGGLE SETUP
# ---------------------------

def setup_kaggle_api_from_file(kaggle_json_path="kaggle.json"):
    """Move kaggle.json to ~/.kaggle and set permissions (Colab-style).
    If running locally, make sure you already have ~/.kaggle/kaggle.json.
    """
    home_kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(home_kaggle_dir, exist_ok=True)
    dest = os.path.join(home_kaggle_dir, "kaggle.json")
    try:
        if os.path.exists(kaggle_json_path):
            subprocess.run(["cp", kaggle_json_path, dest], check=False)
            os.chmod(dest, 0o600)
            print(f"kaggle.json moved to {dest} and permissions set")
        else:
            print(f"kaggle.json not found at {kaggle_json_path}. Please upload it or place it at {dest}.")
    except Exception as e:
        print("Could not set up kaggle.json automatically:", e)


# Example: list some Kaggle datasets (requires kaggle CLI and a valid token)
def example_kaggle_list(search_term="imdb", max_results=5):
    try:
        subprocess.run(["kaggle", "datasets", "list", "-s", search_term, "--max", str(max_results)])
    except Exception as e:
        print("kaggle CLI not available or failed:", e)


# --------------------------------------
# 3) HUGGING FACE PIPELINE DEMOS
# --------------------------------------

def hf_pipeline_demos():
    """Demonstrate Hugging Face pipeline examples: sentiment, translation,
    summarization, and NER.
    """
    try:
        from transformers import pipeline
    except Exception as e:
        print("transformers not installed or failed to import:", e)
        return

    print("\n--- Sentiment Analysis (pipeline) ---")
    sentiment = pipeline("sentiment-analysis")  # default model
    texts = [
        "I loved the new movie! The acting was great and the plot was exciting.",
        "This product was terrible. It broke on day 2 and customer support was unhelpful."
    ]
    results = sentiment(texts)
    for t, r in zip(texts, results):
        print(t)
        print(" ->", r)
        print()

    print("--- Translation (English -> German) ---")
    translator = pipeline("translation_en_to_de")
    text = "Machine learning is changing the world. It's exciting!"
    print(translator(text, max_length=100))

    print("--- Summarization ---")
    summarizer = pipeline("summarization")
    article = (
        "The field of artificial intelligence has seen dramatic progress in recent years. "
        "Large language models now perform many tasks previously believed to require human "
        "intelligence. Researchers continue to push the bounds by scaling models and improving "
        "pretraining and fine-tuning techniques."
    )
    print("Summary:\n", summarizer(article, max_length=60, min_length=20, do_sample=False)[0]['summary_text'])

    print("--- Token classification / NER ---")
    ner = pipeline("ner", grouped_entities=True)
    text = "Barack Obama was born in Hawaii and served as the President of the United States."
    print(ner(text))


# --------------------------------------
# 4) SENTENCE-TRANSFORMERS (EMBEDDINGS)
# --------------------------------------

def sentence_transformers_demo():
    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception as e:
        print("sentence-transformers not installed or failed to import:", e)
        return

    model = SentenceTransformer('all-MiniLM-L6-v2')  # compact, fast
    sentences = [
        "How do I cook pasta?",
        "Best ways to boil spaghetti",
        "What is the capital of France?"
    ]
    embeddings = model.encode(sentences, convert_to_tensor=True)
    print("Embeddings shape:", embeddings.shape)

    query = "How to make spaghetti?"
    q_emb = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(q_emb, embeddings)[0]
    for s, score in zip(sentences, scores):
        print(f"{score.item():.3f}\t{s}")


# --------------------------------------
# 5) FASTAI TEXT CLASSIFICATION EXAMPLE
# --------------------------------------

def fastai_text_classification_demo(use_small_subset=True):
    try:
        from datasets import load_dataset
        from fastai.text.all import TextDataLoaders, text_classifier_learner, AWD_LSTM, accuracy
    except Exception as e:
        print("fastai or datasets not available:", e)
        return

    print("Loading IMDb dataset via Hugging Face datasets...")
    ds = load_dataset("imdb")
    print(ds)

    if use_small_subset:
        small_train = ds['train'].shuffle(seed=42).select(range(4000))   # 4k samples
        small_test  = ds['test'].shuffle(seed=42).select(range(2000))    # 2k samples
    else:
        small_train = ds['train']
        small_test = ds['test']

    import pandas as pd
    train_df = pd.DataFrame(small_train)
    test_df  = pd.DataFrame(small_test)

    dls = TextDataLoaders.from_df(train_df, text_col='text', label_col='label', valid_pct=0.1, bs=32)
    learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
    learn.fine_tune(1)


# --------------------------------------
# 6) HUGGING FACE DATASETS + TOKENIZER
# --------------------------------------

def datasets_and_tokenizer_example():
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except Exception as e:
        print("datasets or transformers not available:", e)
        return

    dataset = load_dataset("glue", "sst2")
    print(dataset['train'][0])

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_fn(batch):
        return tokenizer(batch['sentence'], truncation=True, padding='max_length', max_length=128)

    tokenized = dataset.map(tokenize_fn, batched=True)
    print(tokenized['train'][0])


# --------------------------------------
# 7) RUNNING INFERENCE AND SAVING OUTPUTS
# --------------------------------------

def run_inference_and_save_csv(output_dir="."):
    try:
        from transformers import pipeline
    except Exception as e:
        print("transformers not installed:", e)
        return

    sentiment = pipeline("sentiment-analysis")
    texts = [
        "I love this phone. The battery life is excellent.",
        "Worst purchase ever. It stopped working after two days."
    ]
    results = sentiment(texts)

    out_path = os.path.join(output_dir, "sentiment_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "score"])
        for t, r in zip(texts, results):
            writer.writerow([t, r['label'], r['score']])
    print(f"Saved {out_path}")

    # Example: copy to Drive if running in Colab
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        subprocess.run(["cp", out_path, "/content/drive/MyDrive/"], check=False)
        print("Copied results to /content/drive/MyDrive/")
    except Exception:
        print("Not in Colab or google.colab not available; file left in working directory.")


# --------------------------------------
# 8) PRACTICAL TIPS (as comments in code)
# --------------------------------------
# Token limits: chunk long text for summarization/translation
# GPU: use GPU in Colab for training. Check with !nvidia-smi
# Model size: prefer smaller models (t5-small, distil variants) for quick labs
# Reproducibility: set seeds (PyTorch, numpy, random)
# Kaggle quotas: be mindful of API limits


# --------------------------------------
# MAIN: run selected demos
# --------------------------------------
if __name__ == "__main__":
    print("GenAI-Lab-2 assembled demo script. Choose what to run from the functions provided.")
    
    # Get the directory where this script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Uncomment the actions you want to run locally or in Colab

    install_and_verify_packages()
    
    # Look for kaggle.json in the script directory
    kaggle_json = os.path.join(SCRIPT_DIR, 'kaggle.json')
    setup_kaggle_api_from_file(kaggle_json_path=kaggle_json)
    
    example_kaggle_list()
    hf_pipeline_demos()
    sentence_transformers_demo()
    fastai_text_classification_demo(use_small_subset=True)
    datasets_and_tokenizer_example()
    
    # Pass the script directory to save the CSV there
    run_inference_and_save_csv(output_dir=SCRIPT_DIR)

    print("Code Executed Successfully")
