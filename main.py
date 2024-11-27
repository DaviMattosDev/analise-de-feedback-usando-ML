import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('portuguese'))

def remove_stop_words(text):
    words = text.split()
    return " ".join([word for word in words if word.lower() not in stop_words])

df_positivos = pd.read_csv('positivo.csv')
df_negativos = pd.read_csv('negativo.csv')

comentarios_positivos = df_positivos['frase'].tolist()
comentarios_negativos = df_negativos['frase'].tolist()

rotulos_positivos = [1] * len(comentarios_positivos)
rotulos_negativos = [0] * len(comentarios_negativos)

comentarios = comentarios_positivos + comentarios_negativos
rotulos = rotulos_positivos + rotulos_negativos

comentarios = [remove_stop_words(c) for c in comentarios]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(comentarios)
y = rotulos

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

modelo = MultinomialNB()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acuracia:.2f}")

def tratar_negacao(frase):
    palavras_negativas = ['não', 'nunca', 'odeio', 'detesto', 'horrível', 'triste', 'insuportável', 'péssimo', 'detestei']
    for palavra in palavras_negativas:
        if palavra in frase.lower():
            return 'negativo'  # Se encontrar uma palavra negativa, marca como negativo
    return 'positivo'  # Caso contrário, marca como positivo

def analisar_sentimento(frase):
    frase_preprocessada = remove_stop_words(frase)
    X_new = tfidf.transform([frase_preprocessada])
    previsao = modelo.predict(X_new)
    
    # Verificar se a frase tem palavras de negação
    sentimento_negacao = tratar_negacao(frase)
    
    if sentimento_negacao == 'negativo':
        return "A frase é ruim (negativa)!"
    elif previsao == 1:
        return "A frase é boa (positiva)!"
    else:
        return "A frase é ruim (negativa)!"

# Teste interativo
while True:
    frase = input("Digite uma frase (ou 'sair' para encerrar): ")
    if frase.lower() == 'sair':
        break
    print(analisar_sentimento(frase))
