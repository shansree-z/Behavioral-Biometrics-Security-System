# 🔐 Behavioral Biometrics Security System

> **Your typing rhythm is your password** — Siamese Neural Network authenticates users by how they type, not just what they type.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🧠 How It Works

```
User types phrase → Capture timing events → Extract 40 features
       ↓
Siamese Neural Network compares rhythm against enrolled profile
       ↓
Similarity Score → ACCEPT ✅ or REJECT ❌
```

Even if someone knows your password — they **cannot fake your typing rhythm**.

check it out here!!
🔗 Live Demo (password protected): https://replit.com/@lang99sszkvstss/Behavioral-Biometrics-Security-System

🔑 Password: eployit12345@99


---

## 🚀 Features

- ⌨️ Keystroke dynamics (dwell time, flight time, rhythm ratios)
- 🖱️ Mouse movement analysis
- 🧬 Siamese Neural Network (works with minimal data)
- 📈 Adaptive profile (learns your variations)
- 🌐 Live web interface
- 💾 SQLite persistent storage

---

## 📁 Project Structure

```
biometrics_system/
├── app.py                 # Flask backend + Neural Network
├── requirements.txt       # Dependencies
├── biometrics.db          # Auto-created on first run
├── templates/
│   └── index.html         # Frontend UI
└── README.md
```

---

## ⚙️ Setup & Run

### Local (Laptop)
```bash
git clone https://github.com/YOUR_USERNAME/behavioral-biometrics.git
cd behavioral-biometrics
pip install -r requirements.txt
python app.py
```
Open → `http://localhost:5000`

### Replit
1. Import repo via GitHub URL in Replit
2. Shell → `pip install -r requirements.txt`
3. Change last line of app.py:
```python
app.run(host="0.0.0.0", port=8080)
```
4. Click **Run**

---

## 🎯 Usage

| Step | Action |
|------|--------|
| 1 | Enter username |
| 2 | Go to **Enroll** tab |
| 3 | Type `the quick brown fox` 12+ times |
| 4 | Model trains automatically |
| 5 | Go to **Authenticate** tab |
| 6 | Type once → get similarity score |

---

## 🏗️ Neural Network Architecture

```
Input (40 features)
    ↓
Encoder Branch A ──┐
                   ├── L1 Distance → Dense(8) → Sigmoid
Encoder Branch B ──┘

Each Encoder:
  Dense(64) + BatchNorm + Dropout(0.2)
  Dense(32) + BatchNorm
  Dense(16) → Embedding
```

**Why Siamese?** Only needs YOUR data — no other users required.

---

## 📊 Features Extracted

| Feature | Description |
|---------|-------------|
| Dwell time | How long each key is held |
| Flight time | Gap between keystrokes |
| Rhythm ratios | Inter-key timing ratios (stable across speed) |
| WPM | Typing speed |
| Error rate | Backspace frequency |
| Mouse speed | Cursor movement dynamics |

---

## 🔐 Decision Thresholds

| Score | Decision |
|-------|----------|
| > 72% | ✅ Accept |
| 55–72% | ✅ Accept (monitor) |
| 38–55% | ❌ Reject |
| < 38% | ❌ Reject + Alert |

---

## 🛠️ Tech Stack

- **Backend** — Python, Flask, TensorFlow/Keras, SQLite
- **Frontend** — HTML, CSS, Vanilla JavaScript
- **ML** — Siamese Neural Network, StandardScaler, TF-IDF concepts

---



## 👩‍💻 Author

**Shansree K**
B.Tech AI & Data Science | RP Sarathy Institute of Technology
