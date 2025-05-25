import csv
import os
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.text import Text
import matplotlib.pyplot as plt
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk import word_tokenize
from textwrap import fill
from nltk.corpus import wordnet as wn

# Ruta del archivo CSV
csv_path = "./NPL-Proyecto/codigo/pokedex.csv"

# Leer el CSV y crear un corpus con la feature 'type'
corpus_with_type = []
with open(csv_path, encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        # Cada elemento es un diccionario con la descripción y el tipo
        corpus_with_type.append({
            "info": row["info"],
            "type": row["type"]
        })

# Ejemplo: mostrar las primeras 3 entradas del corpus enriquecido
'''
print("Primeras 3 entradas del corpus con la feature 'type':\n")
for entry in corpus_with_type[:3]:
    print(f"Descripción: {entry['info']}")
    print(f"Tipo(s): {entry['type']}\n")
'''
    
# Si quieres guardar solo las descripciones para análisis de texto:
info_text_path = "info_corpus.txt"
with open(info_text_path, "w", encoding="utf-8") as text_file:
    for entry in corpus_with_type:
        text_file.write(entry["info"] + "\n")

# Crear un corpus con PlaintextCorpusReader (solo descripciones)
corpus_root = os.path.dirname(info_text_path)
corpus = PlaintextCorpusReader(corpus_root, os.path.basename(info_text_path))
text = Text(corpus.words())
fdist = FreqDist(text)
text_str = " ".join(text)
tokens = word_tokenize(text_str)

# Usar como features los tipos de los pokemons
types = ["fire", "water", "grass", "electric", "bug", "normal", "poison", "ghost", "rock", "ground", "fighting", "psychic", "fairy", "dragon", "dark", "ice", "steel", "flying"]

# Usar como features los sinómimos de los tipos de los pokemons
'''
synsets = [syn for poke_type in types for syn in wn.synsets(poke_type)]
types = set()
for syn in synsets:
    for lemma in syn.lemma_names():
        if "_" not in lemma:
            types.add(lemma)

types = list(types)
'''
# Usar como features los hipónimos de los tipos de los pokemons
#types = set(lemma.name() for type in types for synset in wn.synsets(type) for hyp in synset.hyponyms() for lemma in hyp.lemmas() if '_' not in lemma.name())

# Usar como features los hiperónimos de los tipos de los pokemons
#types = set(lemma.name() for type in types for synset in wn.synsets(type) for hyp in synset.hypernyms() for lemma in hyp.lemmas() if '_' not in lemma.name()).union(atypes)

# Ahora tienes corpus_with_type como lista de diccionarios con 'info' y 'type'
# y el corpus de descripciones para análisis textual


import random

# --- Función para extraer características simples de una descripción ---
def description_features(description):
    description_lower = description.lower()
    wnl = nltk.WordNetLemmatizer()
    description_lower = wnl.lemmatize(description_lower)
    features = {}
    for poke_type in types:
        if poke_type in description_lower:
            features[f"has({poke_type})"] = True
    return features

#print(corpus_with_type[0]["info"])
#print(description_features(corpus_with_type[0]["info"]))

# --- Preparamos los datos para el clasificador ---
# Nota: Usamos solo el primer tipo si hay varios (ej: "{fire, flying}" -> "fire")
labeled_features = []
for entry in corpus_with_type:
    desc = entry["info"]
    # Extrae el primer tipo (sin llaves ni espacios)
    main_type = entry["type"].replace("{", "").replace("}", "").split(",")[0].strip()
    feats = description_features(desc)
    labeled_features.append((feats, main_type))
#print(labeled_features)
# Mezclamos y separamos en entrenamiento y prueba (80% train, 20% test)
random.shuffle(labeled_features)
split = int(0.8 * len(labeled_features))
train_set = labeled_features[:split]
test_set = labeled_features[split:]

# --- Entrenamos el clasificador ---
classifier = nltk.NaiveBayesClassifier.train(train_set)

# --- Evaluamos el clasificador ---

media = 0
n = 0
for i in range(0, 400):
    media += nltk.classify.accuracy(classifier, test_set)
    n += 1
accuracy = media / n

print(f"\nAccuracy on test set: {accuracy:.2f}")

#print("\nMost informative features:")
classifier.show_most_informative_features(10)

# --- Ejemplo de predicción ---
print("\nEjemplo de predicción sobre una descripción falsa:")
example = "This Pokémon breathes fire and loves hot places."
features = description_features(example)
predicted_type = classifier.classify(features)
print(f"Descripción: {example}")
print(f"Tipo predicho: {predicted_type}")


from collections import defaultdict

# --- Calcular precisión y recall para cada clase ---
def precision_recall(classifier, test_set, labels):
    # Inicializar contadores
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for feats, true_label in test_set:
        predicted = classifier.classify(feats)
        for label in labels:
            if predicted == label and true_label == label:
                tp[label] += 1
            elif predicted == label and true_label != label:
                fp[label] += 1
            elif predicted != label and true_label == label:
                fn[label] += 1

    # Calcular precisión y recall para cada clase
    for label in labels:
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0
        print(f"Clase: {label}")
        print(f"  Precisión: {precision:.2f}")
        print(f"  Recall:    {recall:.2f}\n")

# Obtener todas las etiquetas posibles del conjunto de prueba
labels = sorted(set(label for _, label in test_set))

print("\nPrecisión y recall por clase:")
precision_recall(classifier, test_set, labels)
