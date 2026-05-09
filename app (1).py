"""
╔══════════════════════════════════════════════════════════════╗
║   BEHAVIORAL BIOMETRICS SECURITY SYSTEM                      ║
║   Keystroke + Mouse Dynamics Authentication                  ║
║   Siamese Neural Network | Flask Backend                     ║
╚══════════════════════════════════════════════════════════════╝

HOW IT WORKS:
  1. User types a fixed phrase 15+ times  → ENROLLMENT
  2. Features extracted from typing rhythm → FEATURE EXTRACTION
  3. Siamese NN trained on your patterns  → TRAINING
  4. New attempt compared to your profile → AUTHENTICATION

RUN:
  pip install -r requirements.txt
  python app.py
  Open: http://localhost:5000
"""

import os, pickle, warnings
import numpy as np
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

TARGET_PHRASE   = "the quick brown fox"
MIN_SAMPLES     = 12      # minimum enrollment samples to train
DB_PATH         = "biometrics.db"

app = Flask(__name__)
app.secret_key = "biometrics_2026"

# ─────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            id        INTEGER PRIMARY KEY,
            username  TEXT,
            features  BLOB,
            timestamp TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username     TEXT PRIMARY KEY,
            profile      BLOB,
            model_data   BLOB,
            scaler_data  BLOB,
            sample_count INTEGER,
            created_at   TEXT
        )
    """)
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# The core insight: extract RHYTHM, not raw speed
# ─────────────────────────────────────────────────────────────

def extract_features(keystrokes, mouse_events):
    """
    From raw timing events → 40-dim behavioral fingerprint.

    Key insight: we extract RATIOS and RELATIVE patterns
    so that fast-typing-you and slow-typing-you produce
    similar vectors. An impostor cannot fake these ratios.
    """
    if not keystrokes or len(keystrokes) < 8:
        return None

    # ── Dwell times (how long each key is held) ──────────────
    dwell = []
    key_events = {}  # key → press timestamp

    for event in keystrokes:
        k   = event.get("key", "")
        t   = event.get("time", 0)
        typ = event.get("type", "")

        if typ == "keydown":
            key_events[k] = t
        elif typ == "keyup" and k in key_events:
            d = t - key_events[k]
            if 10 < d < 1000:          # filter jitter & held keys
                dwell.append(d)
            del key_events[k]

    if len(dwell) < 5:
        return None

    # ── Flight times (gap between consecutive keystrokes) ────
    down_times = [e["time"] for e in keystrokes if e.get("type") == "keydown"]
    flight = []
    for i in range(1, len(down_times)):
        f = down_times[i] - down_times[i - 1]
        if 0 < f < 2000:
            flight.append(f)

    dwell  = np.array(dwell,  dtype=np.float32)
    flight = np.array(flight, dtype=np.float32) if flight else np.array([100.0])

    # ── Statistical features ──────────────────────────────────
    def stats(arr):
        return [
            float(np.mean(arr)),
            float(np.std(arr) + 1e-6),
            float(np.min(arr)),
            float(np.max(arr)),
            float(np.percentile(arr, 25)),
            float(np.percentile(arr, 75)),
        ]

    features = stats(dwell) + stats(flight)

    # ── Rhythm ratios: STABLE across speed variations ─────────
    # If you type "t-h-e" fast or slow, the RATIO t:h:e stays same
    if len(dwell) >= 3:
        ratios = dwell[1:] / (dwell[:-1] + 1e-6)
        ratios = np.clip(ratios, 0.1, 10.0)
        features += [float(np.mean(ratios)), float(np.std(ratios))]
    else:
        features += [1.0, 0.0]

    # ── Per-key timings (first 10 positions, normalized) ──────
    target_len = 10
    norm_dwell = list(dwell[:target_len] / (np.mean(dwell) + 1e-6))
    while len(norm_dwell) < target_len:
        norm_dwell.append(1.0)

    norm_flight = list(flight[:target_len] / (np.mean(flight) + 1e-6))
    while len(norm_flight) < target_len:
        norm_flight.append(1.0)

    features += norm_dwell + norm_flight

    # ── Typing speed ──────────────────────────────────────────
    total_time_min = (down_times[-1] - down_times[0]) / 60000 + 1e-6
    wpm = (len(down_times) / 5) / total_time_min
    features.append(float(np.clip(wpm, 0, 300)))

    # ── Error rate (backspace usage) ─────────────────────────
    backspaces = sum(1 for e in keystrokes
                     if e.get("key") == "Backspace" and e.get("type") == "keydown")
    features.append(float(backspaces / (len(down_times) + 1e-6)))

    # ── Mouse dynamics ────────────────────────────────────────
    if mouse_events and len(mouse_events) >= 5:
        speeds = []
        for i in range(1, len(mouse_events)):
            dx = mouse_events[i]["x"] - mouse_events[i-1]["x"]
            dy = mouse_events[i]["y"] - mouse_events[i-1]["y"]
            dt = mouse_events[i]["t"] - mouse_events[i-1]["t"] + 1e-6
            speeds.append(np.sqrt(dx**2 + dy**2) / dt)
        speeds = np.array(speeds)
        features += [
            float(np.mean(speeds)),
            float(np.std(speeds) + 1e-6),
            float(np.max(speeds)),
        ]
    else:
        features += [0.0, 0.0, 0.0]

    return np.array(features, dtype=np.float32)

# ─────────────────────────────────────────────────────────────
# SIAMESE NEURAL NETWORK
# ─────────────────────────────────────────────────────────────

def build_siamese_network(input_dim):
    """
    Siamese Network: two identical branches share weights.
    Input: two feature vectors (sample_A, sample_B)
    Output: similarity score 0→1
              0 = different person
              1 = same person

    Why Siamese and not a classifier?
    A classifier needs data from many users.
    Siamese only needs YOUR data — it learns
    "are these two samples from the same person?"
    """

    def encoder_branch(input_dim):
        inp = keras.Input(shape=(input_dim,))
        x = layers.Dense(64, activation="relu")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(16, activation="relu")(x)  # embedding
        return keras.Model(inp, x, name="encoder")

    encoder = encoder_branch(input_dim)

    input_a = keras.Input(shape=(input_dim,), name="sample_A")
    input_b = keras.Input(shape=(input_dim,), name="sample_B")

    embed_a = encoder(input_a)
    embed_b = encoder(input_b)

    # L1 distance between embeddings
    distance = layers.Lambda(
        lambda t: tf.abs(t[0] - t[1]),
        name="L1_distance"
    )([embed_a, embed_b])

    x = layers.Dense(8, activation="relu")(distance)
    output = layers.Dense(1, activation="sigmoid", name="similarity")(x)

    model = keras.Model(inputs=[input_a, input_b], outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model, encoder


def create_pairs(samples):
    """
    Build training pairs:
    Positive pairs (label=1): two samples from same user
    Negative pairs (label=0): real sample vs augmented impostor

    Why augment? We only have one user's data.
    We simulate impostors by heavily perturbing the real data.
    A real impostor's timing patterns will look even more different.
    """
    rng = np.random.default_rng(42)
    A, B, labels = [], [], []
    n = len(samples)

    # Positive pairs
    for i in range(n):
        for j in range(i + 1, n):
            A.append(samples[i])
            B.append(samples[j])
            labels.append(1.0)

    # Negative pairs (3× more to balance)
    std = np.std(samples, axis=0) + 1e-6
    for i in range(n):
        for _ in range(3):
            noise    = rng.normal(0, 1.0, samples[i].shape)
            impostor = samples[i] + noise * std * rng.uniform(0.5, 1.5)
            # Randomly permute some features (simulate different person's rhythm)
            idx = rng.choice(len(impostor), size=len(impostor) // 3, replace=False)
            impostor[idx] = rng.uniform(np.min(samples), np.max(samples), len(idx))
            A.append(samples[i])
            B.append(impostor)
            labels.append(0.0)

    A, B, labels = np.array(A), np.array(B), np.array(labels)

    # Shuffle
    perm = rng.permutation(len(labels))
    return A[perm], B[perm], labels[perm]


def train_user_model(samples):
    from sklearn.preprocessing import StandardScaler
    samples   = np.array(samples)
    scaler    = StandardScaler()
    normed    = scaler.fit_transform(samples)
    A, B, y   = create_pairs(normed)
    input_dim = samples.shape[1]

    model, encoder = build_siamese_network(input_dim)
    model.fit(
        [A, B], y,
        epochs=25,
        batch_size=16,
        validation_split=0.15,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True, verbose=0
            )
        ],
    )
    return model, encoder, scaler, input_dim


def compute_auth_score(new_sample, stored_samples, model, scaler):
    """
    Compare new sample against ALL stored enrollment samples.
    Use weighted average: give more weight to highest scores
    (your best typing sessions define your profile).
    """
    new_norm    = scaler.transform([new_sample])
    stored_norm = scaler.transform(stored_samples)

    scores = []
    for s in stored_norm[;8]:
        score = float(model.predict([new_norm, [s]], verbose=0)[0][0])
        scores.append(score)

    scores = np.array(scores)
    # Weighted: avg(70%) + max(30%)
    final = 0.70 * np.mean(scores) + 0.30 * np.max(scores)

    if final >= 0.72:
        decision, level = "ACCEPT", "High"
    elif final >= 0.55:
        decision, level = "ACCEPT", "Medium"
    elif final >= 0.38:
        decision, level = "REJECT", "Low"
    else:
        decision, level = "REJECT", "High"

    return {
        "score":      round(float(final), 4),
        "avg_score":  round(float(np.mean(scores)), 4),
        "max_score":  round(float(np.max(scores)), 4),
        "decision":   decision,
        "confidence": level,
    }

# ─────────────────────────────────────────────────────────────
# FLASK API ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", phrase=TARGET_PHRASE, min_samples=MIN_SAMPLES)


@app.route("/api/enroll", methods=["POST"])
def enroll():
    data       = request.json
    username   = data.get("username", "user").strip()
    keystrokes = data.get("keystrokes", [])
    mouse      = data.get("mouse", [])

    features = extract_features(keystrokes, mouse)
    if features is None:
        return jsonify({"success": False,
                        "message": "Not enough data. Type the full phrase completely."})

    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("INSERT INTO samples (username, features, timestamp) VALUES (?,?,?)",
              (username, pickle.dumps(features), datetime.now().isoformat()))

    c.execute("SELECT COUNT(*) FROM samples WHERE username=?", (username,))
    count = c.fetchone()[0]
    conn.commit()

    trained = False
    if count >= MIN_SAMPLES:
        # Retrain every new sample (rolling retrain)
        c.execute("SELECT features FROM samples WHERE username=?", (username,))
        all_feats = [pickle.loads(r[0]) for r in c.fetchall()]

        model, encoder, scaler, input_dim = train_user_model(all_feats)

        c.execute("""INSERT OR REPLACE INTO users
                     (username, profile, model_data, scaler_data, sample_count, created_at)
                     VALUES (?,?,?,?,?,?)""",
                  (username,
                   pickle.dumps(all_feats),
                   pickle.dumps({"weights": model.get_weights(), "input_dim": input_dim}),
                   pickle.dumps(scaler),
                   count,
                   datetime.now().isoformat()))
        conn.commit()
        trained = True

    conn.close()
    remaining = max(0, MIN_SAMPLES - count)

    return jsonify({
        "success":      True,
        "sample_count": count,
        "trained":      trained,
        "ready":        count >= MIN_SAMPLES,
        "remaining":    remaining,
        "message":      (f"✅ Model trained! {count} samples learned. You can now authenticate."
                         if trained else
                         f"Sample {count} captured. {remaining} more needed to train."),
    })


@app.route("/api/authenticate", methods=["POST"])
def authenticate():
    data       = request.json
    username   = data.get("username", "user").strip()
    keystrokes = data.get("keystrokes", [])
    mouse      = data.get("mouse", [])

    features = extract_features(keystrokes, mouse)
    if features is None:
        return jsonify({"success": False,
                        "message": "Not enough data captured. Type the full phrase."})

    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT profile, model_data, scaler_data FROM users WHERE username=?",
              (username,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({"success": False,
                        "message": f"User '{username}' not enrolled yet."})

    stored_samples = pickle.loads(row[0])
    model_data     = pickle.loads(row[1])
    scaler         = pickle.loads(row[2])

    # Rebuild model from saved weights
    model, _ = build_siamese_network(model_data["input_dim"])
    model.set_weights(model_data["weights"])

    result = compute_auth_score(features, stored_samples, model, scaler)

    return jsonify({"success": True, "username": username, **result})


@app.route("/api/status")
def status():
    username = request.args.get("username", "user").strip()
    conn     = sqlite3.connect(DB_PATH)
    c        = conn.cursor()
    c.execute("SELECT COUNT(*) FROM samples WHERE username=?", (username,))
    count = c.fetchone()[0]
    c.execute("SELECT sample_count FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    return jsonify({
        "sample_count": count,
        "enrolled":     user is not None,
        "ready":        count >= MIN_SAMPLES,
        "remaining":    max(0, MIN_SAMPLES - count),
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    username = request.json.get("username", "user").strip()
    conn     = sqlite3.connect(DB_PATH)
    c        = conn.cursor()
    c.execute("DELETE FROM samples WHERE username=?", (username,))
    c.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": f"Profile for '{username}' cleared."})


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print("=" * 55)
    print("  🔐 Behavioral Biometrics Security System")
    print("  Open: http://localhost:5000")
    print("=" * 55)
    app.run(debug=True, port=5000)
