import csv
import os
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.text import Text
import matplotlib.pyplot as plt
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk import word_tokenize
from textwrap import fill

print("#"*30)
print("Loading corpus ...")
print("#"*30)

# Ruta del archivo CSV
csv_path = "./NPL-Proyecto/codigo/pokedex.csv"

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
text_str = " ".join(text)
tokens = word_tokenize(text_str)

############### Funciones Importantes ###############

# Normalizar el corpus
def normalize_text(text):
    # Tokenizar el texto
    tokens = word_tokenize(text)
    # Convertir a minúsculas
    tokens = [word.lower() for word in tokens]
    # Eliminar puntuación y caracteres no alfabéticos
    tokens = [word for word in tokens if word.isalpha()]
    return tokens

print("\nCorpus loaded successfully.")
print(f"Total number of words in the corpus: {len(text)}")
print(f"Number of unique words in the corpus: {len(set(text))}\n")

print('#'*30)
print('Normalizing the Text ...')
print('#'*30 + '\n')

# Normalizar el corpus
normalized_corpus = normalize_text(text_str)
print(f"Total number of words in the normalized corpus: {len(normalized_corpus)}")

print(normalized_corpus[:20])  # Print first 10 normalized words

print('\n'+'#'*30)
print('Stemming the Text ...')
print('#'*30 + '\n')

porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
porter = [porter.stem(t) for t in normalized_corpus]
lancaster = [lancaster.stem(t) for t in normalized_corpus]

print("Porter Stemmer:")
print(porter[:10])  # Print first 10 stemmed words
print("")
print("Lancaster Stemmer:")
print(lancaster[:10])  # Print first 10 stemmed words
print("")
print(f"Total number of words in the Porter stemmed corpus: {len(porter)}")
print(f"Total number of words in the Lancaster stemmed corpus: {len(lancaster)}")

print('\n'+'#'*30)
print('Lemmatization of the Text ...')
print('#'*30 + '\n')

wnl = nltk.WordNetLemmatizer()
wnl = [wnl.lemmatize(t) for t in porter]

print("WordNet Lemmatizer:")
print(wnl[:10])  # Print first 10 lemmatized words
print("")
print(f"Total number of words in the WordNet lemmatized corpus: {len(wnl)}")

print('\n'+'#'*30)
print('Formatting ...')
print('#'*30 + '\n')

txt = ' '.join(wnl)
txt = fill(txt, width=80)  # Format the text to a specific width
print(txt[:500])  # Print first 1000 characters of the lemmatized text
