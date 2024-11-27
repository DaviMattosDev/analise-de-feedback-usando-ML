import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords

data = pd.read_csv("feedbacks.csv")

modelo_carregado = joblib.load('modelo_feedback.pkl')

# Função para remover stop words e realizar pré-processamento extra
def preprocess_text(text):
    # Converter para minúsculas e remover pontuação
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('portuguese'))
    words = text.split()
    return " ".join([word for word in words if word not in stop_words])

data['Comment'] = data['Comment'].apply(preprocess_text)

predicoes = modelo_carregado.predict(data['Comment'])

data['previsao'] = predicoes

# Mapear os valores das previsões para 'Positivo' (1) e 'Negativo' (0)
data['Sentimento'] = data['previsao'].apply(lambda x: 'Positivo' if x == 1 else 'Negativo')

sentimento_counts = data['Sentimento'].value_counts()

# Criar o gráfico de barras
plt.figure(figsize=(6, 4))
sns.barplot(x=sentimento_counts.index, y=sentimento_counts.values, palette=["green", "red"])
plt.title('Distribuição de Feedbacks')
plt.xlabel('Sentimento')
plt.ylabel('Contagem')
plt.show()

if sentimento_counts.get('Negativo', 0) > sentimento_counts.get('Positivo', 0):
    print("A loja precisa de melhorias! Mais feedbacks negativos que positivos.")
else:
    print("A loja está indo bem, os feedbacks são positivos!")


