import nltk
from nltk import CFG, PCFG, ChartParser, RecursiveDescentParser, ShiftReduceParser, LeftCornerChartParser, DependencyGrammar, parse
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import words as nltk_words
import pandas as pd
from parser import read_csv, Pokemon

# 0. Cargar el corpus de descripciones desde CSV
pokedex_data = read_csv('pokedex.csv')
# Manejar diferentes estructuras retornadas por read_csv
if isinstance(pokedex_data, list) and pokedex_data and isinstance(pokedex_data[0], Pokemon):
    # Lista de objetos Pokemon
    sentences = [str(p.get_info()) for p in pokedex_data]
elif isinstance(pokedex_data, pd.DataFrame):
    sentences = pokedex_data['Info'].astype(str).tolist()
elif isinstance(pokedex_data, list) and isinstance(pokedex_data[0], dict):
    sentences = [str(entry.get('Info', '')) for entry in pokedex_data]
else:
    # Otras estructuras (dict de listas, etc.)
    df_pokedex = pd.DataFrame(pokedex_data)
    sentences = df_pokedex['Info'].astype(str).tolist()

# 1. Ejemplos de frases ambiguas en el corpus
ambiguous = []
for sent in sentences:
    tokens = word_tokenize(sent)
    if any(tok.lower() in {'that', 'and', 'or', 'but'} for tok in tokens):
        ambiguous.append(sent)

print("Ejemplos de oraciones potencialmente ambiguas:")
for s in ambiguous[:5]:
    print('-', s)

# 2. Definir una gramática CFG dinámica que cubra todas las palabras del corpus
# Base de reglas estructurales
grammar_rules = [
    "S -> NP VP",
    "NP -> Det N | Adj N | N | NP PP",
    "VP -> V NP | V NP PP | V",
    "PP -> P NP",
    "Det -> 'the' | 'a' | 'an'",
    "Adj -> 'toxic' | 'poison' | 'ghost' | 'toxicmochi'",
    "P -> 'that' | 'under' | 'to' | 'on' | 'in' | 'with'"
]
# Recolectar el vocabulario del corpus para generar reglas léxicas
vocab = set()
for sent in sentences:
    for w in word_tokenize(sent.lower()):
        # escapa comillas simples en la palabra
        token = w.replace("'", "\'")
        vocab.add(token)
# Generar reglas léxicas: cada token como categoría LEX
grammar_rules.append("LEX -> " + " | ".join(f"'{w}'" for w in vocab))
# Extender producciones para permitir LEX en nominativos y verbos
grammar_rules.append("N -> LEX")
grammar_rules.append("V -> LEX")

# Construir y cargar la gramática CFG
grammar_text = ''.join(grammar_rules)

grammar = CFG.fromstring(grammar_text)

# 3. Probar distintos parsers con la primera frase Probar distintos parsers con la primera frase
sent_test = word_tokenize(sentences[0].lower())
parsers = {
    'RecursiveDescent': RecursiveDescentParser(grammar),
    'ShiftReduce': ShiftReduceParser(grammar),
    'LeftCorner': LeftCornerChartParser(grammar),
    'Chart': ChartParser(grammar)
}
for name, parser in parsers.items():
    try:
        trees = list(parser.parse(sent_test))
        print(f"{name} produce {len(trees)} árboles")
    except Exception as e:
        print(f"{name} error: {e}")

# 4. Construir Well-Formed Substring Table (WFST)
def build_wfst(parser, tokens):
    chart = parser.chart_parse(tokens)
    wf_table = [[False]*len(tokens) for _ in range(len(tokens))]
    for edge in chart.edges():
        s, e = edge.span()
        wf_table[s][e-1] = True
    return wf_table

wfst = build_wfst(parsers['Chart'], sent_test)
print("WFST:")
for row in wfst:
    print(row)

# 5. Dependency Grammar de ejemplo
dep_grammar_text = """
'feeds' -> 'NP' 'NP'
'draw' -> 'NP' 'NP'
"""
dep_grammar = DependencyGrammar.fromstring(dep_grammar_text)
dep_parser = parse.ProjectiveDependencyParser(dep_grammar)
try:
    print(list(dep_parser.parse(sent_test)))
except:
    print("No se pudo parsear dependencias con esta gramática.")

# 6. Ambigüedad vs tamaño de gramática
trees = list(parsers['Chart'].parse(sent_test))
print(f"Producciones CFG: {len(grammar.productions())}, Árboles generados: {len(trees)}")

# 7. Gramática Libre de Contexto Probabilista (PCFG) - probabilidades correctas:
pcfg_text = """
S -> NP VP [1.0]
NP -> Det N [0.25] | Adj N [0.25] | N [0.25] | NP PP [0.25]
VP -> V NP [0.5] | V NP PP [0.3] | V [0.2]
PP -> P NP [1.0]
Det -> 'the' [0.5] | 'a' [0.3] | 'an' [0.2]
Adj -> 'toxic' [0.25] | 'poison' [0.25] | 'ghost' [0.25] | 'toxicmochi' [0.25]
N -> 'mochi' [0.5] | 'desires' [0.5]
V -> 'feeds' [1.0]
P -> 'that' [1.0]
"""
pcfg = PCFG.fromstring(pcfg_text)
pcfg_parser = parse.ViterbiParser(pcfg)
trees_viterbi = list(pcfg_parser.parse(sent_test))
print(f"PCFG Viterbi generó {len(trees_viterbi)} árboles")
