# train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump

# Criar pasta de modelos se não existir
os.makedirs("models", exist_ok=True)

# 🔹 Base de dados simples (spam vs ham)
data = {
    "text": [
        "Parabéns! Você ganhou um prêmio, clique aqui!",
        "Oferta imperdível, compre agora com desconto.",
        "Você foi selecionado para receber um cartão grátis.",
        "Oi, tudo bem? Vamos almoçar amanhã?",
        "Confirme sua presença na reunião de hoje.",
        "Não esqueça de comprar pão na volta para casa.",
        "Seu número foi sorteado, ligue já!",
        "Esse é um lembrete da sua consulta médica amanhã."
    ],
    "label": [
        "spam", "spam", "spam", "ham", "ham", "ham", "spam", "ham"
    ]
}

df = pd.DataFrame(data)

# 🔹 Dividir dados
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# 🔹 Pipeline: TF-IDF + Regressão Logística
pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

# 🔹 Treinar modelo
pipe.fit(X_train, y_train)

# 🔹 Salvar modelo treinado
MODEL_PATH = os.path.join("models", "modelo_spam.pkl")
dump(pipe, MODEL_PATH)

print(f"✅ Modelo treinado e salvo em: {MODEL_PATH}")
