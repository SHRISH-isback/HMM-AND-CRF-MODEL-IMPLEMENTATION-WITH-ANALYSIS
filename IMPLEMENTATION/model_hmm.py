import math
import pickle
from collections import defaultdict, Counter

def default_dict_float():
    return defaultdict(float)

def default_dict_int():
    return defaultdict(int)

class HMMTagger:
    def __init__(self):
        self.transition_probs = defaultdict(default_dict_float)
        self.emission_probs = defaultdict(default_dict_float)
        self.initial_probs = defaultdict(float)
        self.all_tags = set()
        self.transition_counts = defaultdict(default_dict_int)
        self.emission_counts = defaultdict(default_dict_int)
        self.initial_counts = defaultdict(int)
        self.total_sentences = 0
        self.tag_counts = defaultdict(int)
        self.vocabulary = set()
        self.alpha = 0.001

    def train(self, tagged_sentences):
        for sentence in tagged_sentences:
            self.total_sentences += 1
            first_tag = sentence[0][1]
            self.initial_counts[first_tag] += 1
            for word, tag in sentence:
                self.vocabulary.add(word)
                self.all_tags.add(tag)
                self.emission_counts[tag][word] += 1
                self.tag_counts[tag] += 1
            for i in range(len(sentence) - 1):
                self.transition_counts[sentence[i][1]][sentence[i + 1][1]] += 1
        
        for tag in self.initial_counts:
            self.initial_probs[tag] = self.initial_counts[tag] / self.total_sentences
        
        for tag in self.all_tags:
            for word in self.vocabulary:
                count = self.emission_counts[tag].get(word, 0)
                self.emission_probs[tag][word] = (count + self.alpha) / (
                    self.tag_counts[tag] + self.alpha * len(self.vocabulary)
                )
        
        for tag1 in self.all_tags:
            tag1_count = sum(self.transition_counts[tag1].values())
            for tag2 in self.all_tags:
                count = self.transition_counts[tag1].get(tag2, 0)
                self.transition_probs[tag1][tag2] = (count + self.alpha) / (
                    tag1_count + self.alpha * len(self.all_tags)
                )
    
    def predict(self, sentence):
        return self.viterbi_algorithm(sentence)

    def viterbi_algorithm(self, sentence):
        if not sentence:
            return []
        viterbi, backpointer = [{} for _ in range(len(sentence))], [{} for _ in range(len(sentence))]
        
        for tag in self.all_tags:
            word_prob = self.emission_probs[tag].get(sentence[0], self.alpha / 1000)
            viterbi[0][tag] = self.initial_probs.get(tag, 0) * word_prob
            backpointer[0][tag] = None
        
        for t in range(1, len(sentence)):
            for tag in self.all_tags:
                max_prob, best_prev_tag = 0, None
                word_prob = self.emission_probs[tag].get(sentence[t], self.alpha / 1000)
                for prev_tag in self.all_tags:
                    if viterbi[t - 1].get(prev_tag, 0) == 0:
                        continue
                    trans_prob = self.transition_probs[prev_tag].get(tag, self.alpha / 1000)
                    prob = viterbi[t - 1][prev_tag] * trans_prob * word_prob
                    if prob > max_prob:
                        max_prob, best_prev_tag = prob, prev_tag
                viterbi[t][tag], backpointer[t][tag] = max_prob, best_prev_tag
        
        best_last_tag, max_prob = max(viterbi[-1].items(), key=lambda x: x[1], default=(None, 0))
        if best_last_tag is None:
            best_last_tag = list(self.all_tags)[0]
        best_path = [best_last_tag]
        
        for t in range(len(sentence) - 1, 0, -1):
            best_last_tag = backpointer[t].get(best_last_tag, best_path[0])
            best_path.insert(0, best_last_tag)
        
        return best_path

    def handle_unknown_word(self, word):
        return {tag: self.alpha / (self.tag_counts[tag] + self.alpha * len(self.vocabulary)) for tag in self.all_tags}

def read_conllu_file(filepath):
    sentences, current_sentence = [], []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            fields = line.split("\t")
            if len(fields) >= 4 and "-" not in fields[0] and "." not in fields[0]:
                current_sentence.append((fields[1], fields[3]))
    if current_sentence:
        sentences.append(current_sentence)
    return sentences

def read_custom_tags_file(filepath):
    sentences, current_sentence = [], []
    sentence_ending_tags, sentence_ending_symbols = {"PUNC", ".", "?", "!"}, {".", "?", "!"}
    
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                word, tag = parts
                current_sentence.append((word, tag))
                if tag in sentence_ending_tags or any(word.endswith(symbol) for symbol in sentence_ending_symbols):
                    sentences.append(current_sentence)
                    current_sentence = []
    if current_sentence:
        sentences.append(current_sentence)
    return sentences

def map_ud_to_bis(tagged_sentences, mapping=None):
    if mapping is None:
        mapping = {
            "ADJ": "JJ", "ADP": "PSP", "ADV": "RB", "AUX": "VAUX",
            "CCONJ": "CC", "DET": "DT", "INTJ": "INJ", "NOUN": "NN",
            "NUM": "QC", "PART": "RP", "PRON": "PRP", "PROPN": "NNP",
            "PUNCT": "PUNC", "SCONJ": "CC", "SYM": "SYM", "VERB": "VM", "X": "UNK"
        }
    return [[(word, mapping.get(tag, tag)) for word, tag in sentence] for sentence in tagged_sentences]
