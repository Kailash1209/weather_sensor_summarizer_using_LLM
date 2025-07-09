import re
from collections import Counter

class WeatherVocabulary:
    SPECIAL_TOKENS = ["<PAD>", "<START>", "<END>", "<UNK>", "<TEMP>", "<HUM>", "<PRESSURE>"]
    WEATHER_TERMS = {
        "heatwave", "scorching", "brisk", "chilly", "frigid", "humid", "damp",
        "muggy", "stormy", "tropical", "overcast", "drizzling", "blustery",
        "frosty", "sultry", "arid", "monsoon", "cyclone", "sunny", "cloudy",
        "rainy", "windy", "foggy", "hail", "snow", "pleasant", "breezy", "mild",
        "cool", "warm", "hot", "dry", "wet", "clear", "partly", "mostly", "light",
        "moderate", "heavy", "freezing", "icy", "balmy", "crisp", "gale", "gusty",
        "misty", "drizzly", "showers", "thunder", "lightning", "hazy", "smoky",
        "comfortable", "oppressive", "stifling", "torrid", "sweltering", "nippy",
        "arctic", "bitter", "bone-chilling", "wintry", "dank", "clammy",
        "steamy", "temperate", "fair", "gloomy", "dull", "bright", "raw",
        "blistering", "parched"
    }

    def __init__(self, vocab_size=800):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self._build_special_tokens()

    def _build_special_tokens(self):
        for idx, token in enumerate(self.SPECIAL_TOKENS):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'(\d+\.?\d*)\s*°?[CFcf]', r' \1 <TEMP> ', text)
        text = re.sub(r'(\d{1,3})\s*%', r' \1 <HUM> ', text)
        text = re.sub(r'(\d{3,4})\s*(hPa|mb)', r' \1 <PRESSURE> ', text)

        tokens = []
        for word in re.findall(r"\w+(?:[-']\w+)*|['\"]+|[.,!?;]|<[^>]+>|\d+\.?\d*", text):
            tokens.append(word)
        return tokens

    def build(self, texts, min_freq=1):
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        for term in self.WEATHER_TERMS:
            if term not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[term] = idx
                self.idx2word[idx] = term

        for word, freq in counter.most_common():
            if word not in self.word2idx and freq >= min_freq:
                if len(self.word2idx) < self.vocab_size:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
        print(f"Actual vocab size: {len(self.word2idx)}")

    def encode(self, text, max_len=60):  # Increased from 25 to 60
        tokens = self.tokenize(text)
        tokens = [self.SPECIAL_TOKENS[1]] + tokens + [self.SPECIAL_TOKENS[2]]
        if len(tokens) > max_len - 1:
            tokens = tokens[:max_len-1] + [self.SPECIAL_TOKENS[2]]

        indices = [
            self.word2idx.get(token, self.word2idx[self.SPECIAL_TOKENS[3]])
            for token in tokens
        ]
        padding = [self.word2idx[self.SPECIAL_TOKENS[0]]] * (max_len - len(indices))
        return indices + padding

    def decode(self, indices, temp=None, hum=None, pres=None):
        tokens = [
            self.idx2word.get(int(idx), "<UNK>")
            for idx in indices
            if idx != self.word2idx["<PAD>"]
        ]
        tokens = [t for t in tokens if t not in ["<START>", "<END>"]]

        clean_tokens = []
        for token in tokens:
            if token.lower() == "<temp>":
                clean_tokens.append(f"{temp}°C")
            elif token.lower() == "<hum>":
                clean_tokens.append(f"{hum}%")
            elif token.lower() == "<pressure>":
                clean_tokens.append(f"{pres} ")
            else:
                clean_tokens.append(token)

        text = " ".join(clean_tokens)
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
