import pandas as pd
import joblib
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('portuguese'))

# Função para remover stop words e realizar limpeza adicional
def preprocess_text(text):
    # Converter para minúsculas e remover pontuação
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    # Remover stop words
    words = text.split()
    return " ".join([word for word in words if word not in stop_words])

# Carregar os arquivos CSV para o treinamento
df_positivos = pd.read_csv('positivo.csv')
df_negativos = pd.read_csv('negativo.csv')

# Concatenar as frases positivas e negativas
comentarios_positivos = df_positivos['frase'].tolist()
comentarios_negativos = df_negativos['frase'].tolist()

# Criar os rótulos: 1 para positivo, 0 para negativo
rotulos_positivos = [1] * len(comentarios_positivos)
rotulos_negativos = [0] * len(comentarios_negativos)

# Concatenar os dados
comentarios = comentarios_positivos + comentarios_negativos
rotulos = rotulos_positivos + rotulos_negativos

# Pré-processamento dos dados
comentarios = [preprocess_text(c) for c in comentarios]

# Definindo o pipeline com o pré-processamento e o modelo
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  
    ('modelo', MultinomialNB()) 
])

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(comentarios, rotulos, test_size=0.25, random_state=42)

# Treinar o modelo
pipeline.fit(X_train, y_train)

# Avaliar a acurácia do modelo
y_pred = pipeline.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acuracia:.2f}")

# Salvar o modelo e o TfidfVectorizer
#joblib.dump(pipeline, 'modelo_sentimento.pkl')

# Função para verificação da frase
def tratar_negacao(frase):
    palavras_negativas = ['não', 'nunca', 'odeio', 'detesto', 'horrível', 'triste', 'insuportável', 'péssimo', 'detestei']
    palavras_positivas = ['bom', 'ótimo', 'excelente', 'satisfeito', 'feliz', 'maravilhoso', 'incrível', 'recomendo']
    
    # Verificar palavras de negação ou palavras positivas
    for palavra in palavras_negativas:
        if palavra in frase.lower():
            return 'negativo'
    for palavra in palavras_positivas:
        if palavra in frase.lower():
            return 'positivo'
    
    # Caso contrário, considerar a previsão do modelo
    return None

# Função para analisar o sentimento
def analisar_sentimento(frase):
    frase_preprocessada = preprocess_text(frase)
    previsao = pipeline.predict([frase_preprocessada])[0]
    
    # Verificar se a frase tem palavras de negação
    sentimento_negacao = tratar_negacao(frase)
    
    if sentimento_negacao == 'negativo':
        return "A frase é ruim (negativa)!"
    elif previsao == 1:
        return "A frase é boa (positiva)!"
    else:
        return "A frase é ruim (negativa)!"
# While para testar o sistema.
while True:
    frase = input("Digite uma frase (ou 'sair' para encerrar): ")
    if frase.lower() == 'sair':
        break
    print(analisar_sentimento(frase))
