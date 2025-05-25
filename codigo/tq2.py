import csv
import os
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.text import Text
import matplotlib.pyplot as plt
from nltk.probability import FreqDist, ConditionalFreqDist

print("#"*30)
print("Loading corpus ...")
print("#"*30)

# Ruta del archivo CSV
csv_path = "./pokedex.csv"

# Extraer el campo "info" y guardarlo en un archivo de texto
info_text_path = "info_corpus.txt"
with open(csv_path, encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    with open(info_text_path, "w", encoding="utf-8") as text_file:
        for row in reader:
            text_file.write(row["info"] + "\n")

# Crear un corpus con PlaintextCorpusReader
corpus_root = os.path.dirname(info_text_path)

############### Variables Importantes ###############
corpus = PlaintextCorpusReader(corpus_root, os.path.basename(info_text_path))
text = Text(corpus.words())
fdist = FreqDist(text)

############### Funciones Importantes ###############

def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word, end=' ')
        word = cfdist[word].max()

print("\nCorpus loaded successfully.")
print(f"Total number of words in the corpus: {len(text)}")
print(f"Number of unique words in the corpus: {len(set(text))}")

print("This Corpus is only abailable in Inglish\n")

print("This Corpus is made of independent, categorized texts, base on the pokedex.")
print("The texts are not related to each other, but they are all about the same topic: Pokemon\n")

print("Tabulatin Distributions\n")
# Creamos una ConditionalFreqDist más interesante:
# Analizaremos la frecuencia de palabras según su longitud para las 200 palabras más frecuentes del corpus

# Obtenemos las 200 palabras más frecuentes
top_words = [word for word, _ in fdist.most_common(200)]

# Creamos la ConditionalFreqDist: condición = longitud de palabra, evento = palabra
cfd = ConditionalFreqDist(
    (len(word), word) for word in corpus.words() if word in top_words
)

print("Frequency of the most common words grouped by word length (top 200 words):")
# Esto ayuda a entender el estilo y la riqueza léxica del corpus, así como posibles patrones de uso de palabras clave
cfd.tabulate(conditions=range(3, 13), samples=top_words[:15])
print("\nThis helps to understand the style and lexical richness of the corpus, as well as possible patterns of keyword usage")

print("\n")
print("Generating Random Text with Bigrams")

bigrams = nltk.bigrams(corpus.words())
cfd = ConditionalFreqDist(bigrams)
generate_model(cfd, 'The')
