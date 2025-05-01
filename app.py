from flask import Flask, render_template, request, redirect, url_for
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import psycopg2
import os
import struct
import gdown

app = Flask(__name__)
app.secret_key = "secret-key-change-this"

# 📥 Télécharger le modèle
def download_model_from_drive():
    if not os.path.exists("model_camembert_final"):
        print("🔽 Téléchargement du modèle depuis Google Drive...")
        os.makedirs("model_camembert_final", exist_ok=True)
        folder_id = "1B2Nriruvr4lpWs7OA7Gyekk80JPwkTcK?usp=sharing"
        gdown.download_folder(
            url=f"https://drive.google.com/drive/folders/{folder_id}",
            output="model_camembert_final",
            quiet=False,
            use_cookies=False
        )

download_model_from_drive()

# 🔄 Charger modèle
model = CamembertForSequenceClassification.from_pretrained("model_camembert_final")
tokenizer = CamembertTokenizer.from_pretrained("model_camembert_final")
model.eval()

# 🔌 Connexion à PostgreSQL
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_connection():
    return psycopg2.connect(DATABASE_URL)

# 🛠️ Initialisation de la table
def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            pseudo TEXT,
            texte TEXT,
            label TEXT,
            proba_misogyne REAL,
            proba_non_misogyne REAL
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

init_db()

def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1).detach().numpy()[0]
    label = logits.argmax(dim=1).item()
    return label, probs

# ✅ Conversion sûre
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
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO messages (pseudo, texte, label, proba_misogyne, proba_non_misogyne)
                VALUES (%s, %s, %s, %s, %s)
            """, (pseudo, texte, label_str, float(probs[1]), float(probs[0])))
            conn.commit()
            cur.close()
            conn.close()
            return redirect(url_for("index"))

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT pseudo, texte, label, proba_misogyne, proba_non_misogyne FROM messages ORDER BY id DESC")
    raw_messages = cur.fetchall()
    cur.close()
    conn.close()

    messages = []
    for pseudo, texte, label, p1, p0 in raw_messages:
        messages.append((pseudo, texte, label, safe_float(p1), safe_float(p0)))

    return render_template("index.html", messages=messages)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
