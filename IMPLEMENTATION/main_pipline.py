import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from collections import defaultdict, Counter

from model_hmm import HMMTagger, read_conllu_file, read_custom_tags_file, map_ud_to_bis
from model_crf import CRFTagger

TRAIN_MODELS = True
TEST_MODELS = True
MODEL_DIR = "pkl_files"
OUTPUT_DIR = "output"
TRAIN_FILE = "hindi_and_eng.conllu"
ENG_TEST_FILE = "eng.txt"
hindi_TEST_FILE = "hindi.txt"

def tokenize_sentence(sentence):
    import re
    tokens = [word for word in re.findall(r"\b\w+\b|[^\w\s]", sentence)]
    return tokens

def prepare_training_data(sentences, split_ratio=0.8):
    random.shuffle(sentences)
    split_idx = int(len(sentences) * split_ratio)
    return sentences[:split_idx], sentences[split_idx:]

def extract_words_tags(tagged_sentences):
    words, tags = [], []
    for sentence in tagged_sentences:
        sentence_words, sentence_tags = zip(*sentence)
        words.append(list(sentence_words))
        tags.append(list(sentence_tags))
    return words, tags

def evaluate_model(true_tags, pred_tags, all_tags):
    true_flat = [tag for sent in true_tags for tag in sent] if isinstance(true_tags[0], list) else true_tags
    pred_flat = [tag for sent in pred_tags for tag in sent] if isinstance(pred_tags[0], list) else pred_tags
    precision, recall, f1, _ = precision_recall_fscore_support(true_flat, pred_flat, average="weighted", zero_division=0)
    accuracy = sum(1 for t, p in zip(true_flat, pred_flat) if t == p) / len(true_flat)
    cm = confusion_matrix(true_flat, pred_flat, labels=all_tags)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "confusion_matrix": cm}

def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    cm_for_display = np.nan_to_num(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]) if normalize else cm
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_for_display, annot=True, fmt=".2f" if normalize else "d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    return plt.gcf()

def plot_tag_distribution(tagged_sentences, title="Tag Distribution"):
    all_tags = [tag for sentence in tagged_sentences for _, tag in sentence]
    if not all_tags:
        plt.figure(figsize=(12, 6))
        plt.title(f"{title} - NO DATA")
        return plt.gcf()
    tag_counts = Counter(all_tags)
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    plt.figure(figsize=(12, 6))
    plt.bar([item[0] for item in sorted_tags], [item[1] for item in sorted_tags])
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("POS Tags")
    plt.ylabel("Frequency")
    plt.tight_layout()
    return plt.gcf()

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hmm_model, crf_model = None, None
    if TRAIN_MODELS:
        hmm_model, crf_model = HMMTagger(), CRFTagger()
        sentences = map_ud_to_bis(read_conllu_file(TRAIN_FILE))
        train_data, val_data = prepare_training_data(sentences, 0.9)
        hmm_model.train(train_data)
        crf_model.train(train_data)
        pickle.dump(hmm_model, open(os.path.join(MODEL_DIR, "hmm_model.pkl"), "wb"))
        pickle.dump(crf_model, open(os.path.join(MODEL_DIR, "crf_model.pkl"), "wb"))
    if TEST_MODELS:
        if not hmm_model or not crf_model:
            hmm_model = pickle.load(open(os.path.join(MODEL_DIR, "model_checkpoints_hmm.pkl"), "rb"))
            crf_model = pickle.load(open(os.path.join(MODEL_DIR, "model_checkpoints_crf.pkl"), "rb"))
        for test_file, lang in [(ENG_TEST_FILE, "English"), (hindi_TEST_FILE, "Hindi")]:
            sentences = read_custom_tags_file(test_file)
            plot_tag_distribution(sentences, title=f"{lang} Test Data - POS Tag Distribution").savefig(os.path.join(OUTPUT_DIR, f"{lang.lower()}.png"))
            words, true_tags = extract_words_tags(sentences)
            hmm_pred_tags = [hmm_model.predict(sentence) for sentence in words]
            crf_pred_tags = [crf_model.predict(sentence) for sentence in words]
            all_tags = list(hmm_model.all_tags)
            hmm_results = evaluate_model(true_tags, hmm_pred_tags, all_tags)
            crf_results = evaluate_model(true_tags, crf_pred_tags, all_tags)
            plot_confusion_matrix(hmm_results["confusion_matrix"], all_tags, normalize=True, title=f"HMM Confusion Matrix ({lang})").savefig(os.path.join(OUTPUT_DIR, f"{lang.lower()}_hmm_confusion_matrix.png"))
            plot_confusion_matrix(crf_results["confusion_matrix"], all_tags, normalize=True, title=f"CRF Confusion Matrix ({lang})").savefig(os.path.join(OUTPUT_DIR, f"{lang.lower()}_crf_confusion_matrix.png"))
main()
