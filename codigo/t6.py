import nltk
from nltk import CFG, PCFG, ChartParser, RecursiveDescentParser, ShiftReduceParser, LeftCornerChartParser, DependencyGrammar, parse
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import words as nltk_words
import pandas as pd
from parser import read_csv, Pokemon
import re

# 0. Load the description corpus from CSV
pokedex_data = read_csv('pokedex.csv')
# Handle different structures returned by read_csv
if isinstance(pokedex_data, list) and pokedex_data and isinstance(pokedex_data[0], Pokemon):
    # List of Pokemon objects
    sentences = [str(p.get_info()) for p in pokedex_data]
elif isinstance(pokedex_data, pd.DataFrame):
    sentences = pokedex_data['Info'].astype(str).tolist()
elif isinstance(pokedex_data, list) and isinstance(pokedex_data[0], dict):
    sentences = [str(entry.get('Info', '')) for entry in pokedex_data]
else:
    # Other structures (dict of lists, etc.)
    df_pokedex = pd.DataFrame(pokedex_data)
    sentences = df_pokedex['Info'].astype(str).tolist()

# Examples of potentially ambiguous sentences in the corpus
ambiguous = []
for sent in sentences:
    tokens = word_tokenize(sent)
    if any(tok.lower() in {'that', 'and', 'or', 'but'} for tok in tokens):
        ambiguous.append(sent)

print("Examples of potentially ambiguous sentences:")
for s in ambiguous[:5]:
    print('-', s)

# Define a dynamic CFG grammar covering all corpus words
# Base structural rules
grammar_rules = [
    "S -> NP VP",
    "NP -> Det N | Adj N | N | NP PP",
    "VP -> V NP | V NP PP | V",
    "PP -> P NP",
    "Det -> 'the' | 'a' | 'an'",
    "Adj -> 'toxic' | 'poison' | 'ghost' | 'toxicmochi'",
    "P -> 'that' | 'under' | 'to' | 'on' | 'in' | 'with'"
]

# ---Collect vocabulary and filter only alphabetic words ---
vocab = set()
for sent in sentences:
    for w in word_tokenize(sent.lower()):
        vocab.add(w)

# Filter: only alphabetic strings (adjust pattern if apostrophes are needed)
filtered_vocab = {w for w in vocab if re.fullmatch(r"[a-zA-Z]+", w)}
if not filtered_vocab:
    raise ValueError("No valid tokens after filtering punctuation and numbers.")

# --- Generate the LEX rule with filtered vocabulary ---
lex_rule = "LEX -> " + " | ".join(f"'{w}'" for w in sorted(filtered_vocab))
grammar_rules.append(lex_rule)
grammar_rules.append("N -> LEX")
grammar_rules.append("V -> LEX")

# ---Build the grammar text with line breaks ---
grammar_text = "\n".join(grammar_rules)

# For debugging, print the grammar text:
print("=== grammar_text ===")
print(grammar_text)

# ---Attempt to create the grammar and display error if it fails ---
try:
    grammar = CFG.fromstring(grammar_text)
except Exception as e:
    print("Error parsing CFG grammar:", e)
    raise  # to see full stack trace on failure


sent_test = word_tokenize(sentences[1024].lower())  # use the last sentence as example

# Probabilistic Context-Free Grammar (PCFG) - with proper probabilities:
pcfg_text = """
S -> NP VP [0.8] | S Conj S [0.2]
NP -> Det N [0.15] | Adj N [0.15] | N [0.15] | NP PP [0.15] | NP Conj NP [0.4]
VP -> V NP [0.4] | V NP PP [0.2] | V [0.1] | VP Conj VP [0.3]
PP -> P NP [1.0]
Det -> 'the' [0.1667] | 'a' [0.1667] | 'an' [0.1667] | 'it' [0.1667] | 'others' [0.1667] | 'its' [0.1667]
Adj -> 'toxic' [0.25] | 'poison' [0.25] | 'ghost' [0.25] | 'toxicmochi' [0.25]
N -> 'mochi' [0.111] | 'desires' [0.111] | 'capabilities' [0.111] | 'those' [0.111] | 'pecharunt' [0.111] | 'control' [0.111] | 'will' [0.111] | 's' [0.111] | 'who' [0.111]
V -> 'feeds' [0.1667] | 'draw' [0.1667] | 'eat' [0.1667] | 'fall' [0.1667] | 'chained' [0.1667] | 'out' [0.1667]
P -> 'that' [0.3333] | 'under' [0.3333] | 'to' [0.3333]
Conj -> 'and' [1.0]
"""
pcfg = PCFG.fromstring(pcfg_text)
pcfg_parser = parse.ViterbiParser(pcfg)
tokens_pcfg = [w for w in sent_test if w.isalpha()]
print("Tokens for PCFG:", tokens_pcfg)
if tokens_pcfg:
    trees_v = list(pcfg_parser.parse(tokens_pcfg))
    print(f"PCFG Viterbi generated {len(trees_v)} trees")
else:
    print("No tokens compatible with the PCFG.")


