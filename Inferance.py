import torch
import numpy as np
import pickle
import re
import random
import time
from sklearn.metrics.pairwise import cosine_similarity
from vocab import WeatherVocabulary
from transformer_model import WeatherTransformer
from read_sensors import read_environment
from gui_display import WeatherDisplay

class Config:
    d_model = 128
    nhead = 4
    num_layers = 2
    max_len = 60
    model_path = "best_weather_model.pth"
    vocab_path = "vocab.pkl"
    stats_path = "stats.pkl"
    temperature = 1.2
    top_k = 60
    repetition_penalty = 2.0
    diversity_penalty = 0.8
    min_length = 15
    max_attempts = 5
    seed = None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clean_summary(text):
    text = re.sub(r"(\\d+\\.?\\d*\\s*°C)(?:\\s+\\1)+", r"\\1", text, flags=re.IGNORECASE)
    text = re.sub(r"(\\d+\\.?\\d*\\s*%)(?:\\s+\\1)+", r"\\1", text, flags=re.IGNORECASE)
    text = re.sub(r"(\\d+\\s*hPa)(?:\\s+\\1)+", r"\\1", text, flags=re.IGNORECASE)
    words = text.split()
    cleaned = []
    for word in words:
        if not cleaned or word != cleaned[-1]:
            cleaned.append(word)
    text = " ".join(cleaned)
    text = re.sub(r"(\d+\.?\d*\s*°C)(?:\s+\1)+", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+\.?\d*\s*%)(?:\s+\1)+", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+\s*hPa)(?:\s+\1)+", r"\1", text, flags=re.IGNORECASE)

    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    return text[0].upper() + text[1:] if text else text

def calculate_similarity(text1, text2):
    words = set(text1.split() + text2.split())
    vec1 = [text1.split().count(w) for w in words]
    vec2 = [text2.split().count(w) for w in words]
    return cosine_similarity([vec1], [vec2])[0][0]

def sample_sequence(model, vocab, sensor_tensor, device, config, temp, hum, pres, previous_summaries=[]):
    tokens = [vocab.word2idx['<START>']]
    length_target = random.randint(config.min_length, config.max_len)

    for i in range(config.max_len):
        token_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(token_tensor, sensor_tensor)[0, -1]
        for token_id in set(tokens):
            if logits[token_id] > 0:
                logits[token_id] /= config.repetition_penalty
            else:
                logits[token_id] *= config.repetition_penalty
        logits = logits / config.temperature
        if previous_summaries and i > 5:
            current_text = vocab.decode(tokens, temp, hum, pres)
            for prev in previous_summaries:
                similarity = calculate_similarity(current_text, prev)
                if similarity > 0.6:
                    logits = logits * (1 - config.diversity_penalty * similarity)
        top_k = min(config.top_k, logits.size(-1))
        top_logits, top_indices = torch.topk(logits, top_k)
        probs = torch.softmax(top_logits, dim=-1)
        next_token = top_indices[torch.multinomial(probs, 1)].item()
        tokens.append(next_token)
        if next_token == vocab.word2idx['<END>'] and len(tokens) >= length_target:
            break
    return clean_summary(vocab.decode(tokens, temp=temp, hum=hum, pres=pres))

def generate_diverse_summary(model, vocab, sensor_tensor, device, config, temp, hum, pres):
    summaries = []
    similarity_threshold = 0.7
    for attempt in range(config.max_attempts):
        seed = random.randint(0, 100000)
        set_seed(seed)
        summary = sample_sequence(model, vocab, sensor_tensor, device, config,
                                  temp, hum, pres, summaries)
        is_diverse = all(calculate_similarity(summary, prev) <= similarity_threshold for prev in summaries)
        if is_diverse or attempt == config.max_attempts - 1:
            return summary
        summaries.append(summary)
    return summary

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    with open(Config.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print(f"📖 Loaded vocabulary with {len(vocab.word2idx)} tokens")

    with open(Config.stats_path, 'rb') as f:
        stats = pickle.load(f)
    print("📊 Loaded normalization stats")

    model = WeatherTransformer(
        vocab_size=len(vocab.word2idx),
        d_model=Config.d_model,
        nhead=Config.nhead,
        num_layers=Config.num_layers
    ).to(device)
    model.load_state_dict(torch.load(Config.model_path, map_location=device))
    model.eval()
    print("🧠 Model loaded successfully")

    display = WeatherDisplay()

    while True:
        reading = read_environment()
        if reading:
            temp, hum, pres = reading
            print(f"\\n📡 Reading: Temp={temp}°C, Hum={hum}%, Pres={pres} hPa")

            norm = np.array([
                (temp - stats['temp'][0]) / (stats['temp'][1] - stats['temp'][0] + 1e-8),
                (hum - stats['hum'][0]) / (stats['hum'][1] - stats['hum'][0] + 1e-8),
                (pres - stats['pres'][0]) / (stats['pres'][1] - stats['pres'][0] + 1e-8),
            ])
            sensor_tensor = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).to(device)

            summary = generate_diverse_summary(model, vocab, sensor_tensor, device, Config, temp, hum, pres)
            print(f"📝 Summary: {summary}")
            display.update(temp, hum, pres, summary)
        else:
            print("❌ Sensor read failed.")

        time.sleep(10)

if __name__ == "__main__":
    main()
