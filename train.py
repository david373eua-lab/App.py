
# train.py
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from joblib import dump

# Criar dados de exemplo (pode trocar depois por um dataset maior)
data = {
    "mensagem": [
        "Parabéns! Você ganhou um prêmio, clique aqui para resgatar!",
        "Oi, tudo bem? Vamos almoçar hoje?",
        "Você foi selecionado para um empréstimo incrível. Ligue agora!",
        "Não esqueça da reunião amanhã às 10h.",
        "Oferta exclusiva: compre já seu cartão de crédito premium!"
    ],
    "rotulo": ["spam", "ham", "spam", "ham", "spam"]
}

df = pd.DataFrame(data)

# Modelo: pipeline (vetorizador + Naive Bayes)
pipe = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

# Treina
pipe.fit(df["mensagem"], df["rotulo"])

# Salvar modelo treinado
os.makedirs("models", exist_ok=True)
dump(pipe, os.path.join("models", "modelo_spam.pkl"))

print("✅ Modelo treinado e salvo em 'models/modelo_spam.pkl'")
