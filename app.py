from flask import Flask, render_template, request, redirect, url_for
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import sqlite3
import struct
import os
import gdown

app = Flask(__name__)
app.secret_key = "secret-key-change-this"

# 📥 Télécharger automatiquement le modèle depuis Google Drive
def download_model_from_drive():
    if not os.path.exists("model_camembert_final"):
        print("🔽 Téléchargement du modèle depuis Google Drive...")
        os.makedirs("model_camembert_final", exist_ok=True)

        # Remplace cet ID par l’ID réel de ton dossier Google Drive partagé
        folder_id = "1cIcTUQBbXiOlVDWrjfOLpa1w3s8zHz7J?usp=sharing"
        gdown.download_folder(
            url=f"https://drive.google.com/drive/folders/{folder_id}",
            output="model_camembert_final",
            quiet=False,
            use_cookies=False
        )

download_model_from_drive()

# 🔄 Charger le modèle après téléchargement
model = CamembertForSequenceClassification.from_pretrained("model_camembert3")
tokenizer = CamembertTokenizer.from_pretrained("model_camembert3")
model.eval()

DB_PATH = "forum.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS messages (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     pseudo TEXT,
                     texte TEXT,
                     label TEXT,
                     proba_misogyne REAL,
                     proba_non_misogyne REAL)''')
        conn.commit()

init_db()

def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1).detach().numpy()[0]
    label = logits.argmax(dim=1).item()
    return label, probs

# ✅ Conversion sécurisée
def safe_float(val):
    if isinstance(val, bytes):
        return struct.unpack('f', val)[0]
    return float(val)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pseudo = request.form["pseudo"].strip()
        texte = request.form["texte"].strip()
        if pseudo and texte:
            label, probs = predict_text(texte)
            label_str = "Misogyne" if label == 1 else "Non Misogyne"
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute("INSERT INTO messages (pseudo, texte, label, proba_misogyne, proba_non_misogyne) VALUES (?, ?, ?, ?, ?)",
                          (pseudo, texte, label_str, float(probs[1]), float(probs[0])))
                conn.commit()
            return redirect(url_for("index"))

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT pseudo, texte, label, proba_misogyne, proba_non_misogyne FROM messages ORDER BY id DESC")
        raw_messages = c.fetchall()

        messages = []
        for pseudo, texte, label, p1, p0 in raw_messages:
            messages.append((pseudo, texte, label, safe_float(p1), safe_float(p0)))

    return render_template("index.html", messages=messages)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
