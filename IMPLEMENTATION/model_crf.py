import sklearn_crfsuite
from sklearn_crfsuite import metrics
from collections import Counter

class CRFTagger:
    def __init__(self, algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100):
        self.model = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
        )
        self.tag_counter = Counter()

    def extract_features(self, sentence, index):
        word = sentence[index]
        features = {
            "bias": 1.0,
            "word.lower()": word.lower(),
            "word[-3:]": word[-3:],
            "word[-2:]": word[-2:],
            "word.isupper()": word.isupper(),
            "word.istitle()": word.istitle(),
            "word.isdigit()": word.isdigit(),
            "word.contains_hyphen": "-" in word,
            "word.contains_digit": any(char.isdigit() for char in word),
        }
        
        for i in range(1, 5):
            if len(word) >= i:
                features[f"prefix-{i}"] = word[:i]
                features[f"suffix-{i}"] = word[-i:]
        
        if index > 0:
            prev_word = sentence[index - 1]
            features.update({
                "prev_word.lower()": prev_word.lower(),
                "prev_word.istitle()": prev_word.istitle(),
                "prev_word[-3:]": prev_word[-3:],
            })
        else:
            features["BOS"] = True
        
        if index < len(sentence) - 1:
            next_word = sentence[index + 1]
            features.update({
                "next_word.lower()": next_word.lower(),
                "next_word.istitle()": next_word.istitle(),
                "next_word[-3:]": next_word[-3:],
            })
        else:
            features["EOS"] = True
        
        return features

    def prepare_data(self, tagged_sentences):
        X, y = [], []
        for sentence in tagged_sentences:
            words, tags = zip(*sentence)
            self.tag_counter.update(tags)
            sentence_features = [self.extract_features(words, i) for i in range(len(words))]
            X.append(sentence_features)
            y.append(tags)
        return X, y

    def train(self, tagged_sentences):
        X, y = self.prepare_data(tagged_sentences)
        self.model.fit(X, y)

    def predict(self, sentence):
        features = [self.extract_features(sentence, i) for i in range(len(sentence))]
        return self.model.predict([features])[0]
