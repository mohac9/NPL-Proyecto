import csv
import os
from nltk.corpus import PlaintextCorpusReader
from nltk.text import Text
import matplotlib.pyplot as plt
from nltk.probability import FreqDist

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

def lexical_diversity(text):
    """
    Calculate the lexical diversity of a text.
    Lexical diversity is defined as the ratio of unique words to total words.
    """
    return len(set(text)) / len(text)

def percentage(count, total):
    """
    Calculate the percentage of a count relative to a total.
    """
    return 100 * count / total

############### Operaciones sobre el código ###############

print("\nCorpus loaded successfully.")
print(f"Total number of words in the corpus: {len(text)}")
print(f"Number of unique words in the corpus: {len(set(text))}")
print(f"Lexical diversity: {lexical_diversity(text):.4f}")

print("=" * 30)
print("Concordance for the word 'plant':")
text.concordance("plant")
print("=" * 30)

print("\nConcordance for the word 'monster':")
text.concordance("monster")
print("=" * 30)

print("\nWords similar to 'monster':")
text.similar("monster")
print("=" * 30)

print("\nCommon contexts between 'fire' and 'water':")
text.common_contexts(["fire", "water"])
print("=" * 30)

print("\nDisplaying Pokémon types dispersion plot...")
text.dispersion_plot(["fire", "water", "grass", "electric", "bug", "normal", "poison", "ghost", "rock", "ground", "fighting", "psychic", "fairy", "dragon", "dark", "ice", "steel", "flying"])
plt.show()

print("\nGenerating random text based on the corpus:")
text.generate()
print("=" * 30)

print("\nWord frequency in the corpus:")
# Frecuencia de palabras
# fdist = FreqDist(text)
print(fdist)
print("\nThe 40 most frequent words:")
print(fdist.most_common(40))
print("\nDisplaying cumulative frequency plot of the 50 most common words...")
fdist.plot(50, cumulative=True)
plt.show()

print("\nWords longer than 10 letters (sorted alphabetically):")
# Conjunto alfabeticamente ordenado de palabras de más de 10 letras
V = set(text)
long_words = [w for w in V if len(w) > 10]
long_words = sorted(long_words)
print(long_words)

print("\nWords longer than 7 letters and appearing more than 7 times (sorted alphabetically):")
# Ordena alfabeticamente el conjunto de palabras de más de 7 letras y que aparecen más de 7 veces en el texto
print(sorted(w for w in set(text) if len(w) > 7 and fdist[w] > 7))

print("\nPairs of words that unusually often appear together (collocations):")
# Pares de palabras que aparecen inusualmente amenudo juntas
text.collocations()

print("\nMost common word length in the text:")
# Número de letras más comun entre las palabras dedl texto
fdist = FreqDist(len(w) for w in text)
print(fdist.most_common())

