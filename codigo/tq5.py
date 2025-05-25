import nltk
import parser as ps
from nltk.probability import FreqDist

# Get the description of the pokemons
descriptions = {}
pokemons = ps.read_csv()


#Tagging the descriptions
#The discriptions are the only camp tagged because they are prhases
tagged_descriptions = {}
for pokemon in pokemons:
    description = pokemon.get_info()
    tagged_description = nltk.pos_tag(nltk.word_tokenize(description))
    tagged_descriptions[pokemon.get_id()] = tagged_description

# Print the tagged descriptions
'''
for pokemon_id, tagged_description in tagged_descriptions.items():
    print(f"ID: {pokemon_id}, Tagged Description: {tagged_description}")
'''
#Search untagged words
def search_untagged_words(tagged_descriptions):
    untagged_words = []
    for pokemon_id, tagged_description in tagged_descriptions.items():
        for word, tag in tagged_description:
            if tag == None:
                untagged_words.append(word)
    return set(untagged_words)

# Get the untagged words
untagged_words = search_untagged_words(tagged_descriptions)
# Print the untagged words
print("Untagged Words:")
for word in untagged_words:
    print(word)
    
print("\nTotal untagged words:", len(untagged_words))
print("Total tagged words:", sum(len(tagged_description) for tagged_description in tagged_descriptions.values()))


##Creation of varius taggers
#Default tagger
default_tagger = nltk.DefaultTagger('NN')
# Regular expression tagger
regexp_tagger = nltk.RegexpTagger(
    [
        (r'.*ing$', 'VBG'), # Gerunds
        (r'.*ed$', 'VBD'),  # Past tense verbs
        (r'.*es$', 'VBZ'),  # 3rd singular present
        (r'.*ould$', 'MD'), # modals
        (r'.*\'s$', 'NN$'), # possessive nouns
        (r'.*s$', 'NNS'), # plural nouns
        (r'^-?[0-9]+(\.[0-9]+)?$', 'CD') , # cardinal numbers
        (r'\b(?:a|an|the)\b', 'DT'), # determiners
        (r'\b(?:I|you|he|she|it|we|they)\b', 'PRP'), # pronouns
        (r'\b(?:am|is|are|was|were|be|being|been)\b', 'VB'), # forms of "to be"
        (r'\b(?:do|does|did|doing|done)\b', 'VB'), # forms of "to do"
        (r'\b(?:will|shall|can|could|may|might|must)\b', 'MD'), # modals
        (r'\b(?:and|or|but|so|for|nor|yet)\b', 'CC'), # conjunctions
        (r'\b(?:if|unless|because|although|while|when)\b', 'IN'), # subordinating conjunctions
        (r'\b(?:this|that|these|those)\b', 'DT'), # demonstratives
        (r'\b(?:here|there|where|when|why|how)\b', 'WRB'), # wh-words
        (r'\b(?:yes|no|maybe|perhaps)\b', 'UH'), # interjections
        (r'\b(?:more|most|less|least)\b', 'RBR'), # comparative and superlative adverbs
        (r'\b(?:very|quite|somewhat|rather)\b', 'RB'), # adverbs
        (r'\b(?:good|bad|happy|sad|angry|excited)\b', 'JJ'), # adjectives
        (r'\b(?:like|love|hate|prefer|enjoy)\b', 'VB'), # verbs of preference
        (r'\b(?:pokemon|pokedex|battle|trainer|gym|league|evolution)\b', 'NN'), # Specific nouns
        (r'\b(?:catch|train|battle|evolve|fight|defeat)\b', 'VB'), # Verbs related to pokemon
        (r'.*', 'NN'), # Default to noun


    ]
)
# Lookup tagger
#Obtain information
all_words = []
for pokemon in pokemons:
    description = pokemon.get_info()
    tokens = nltk.word_tokenize(description)
    all_words.extend(tokens)

# Create frequency distribution
fdist = FreqDist(all_words)
cfd = nltk.ConditionalFreqDist((word.lower(), tag) for (word, tag) in nltk.pos_tag(all_words))
most_freq_words = fdist.most_common(100)
likely_tags = {}
for word, _ in most_freq_words:
    word_lower = word.lower()
    
    if word_lower in cfd and len(cfd[word_lower]) > 0:
        likely_tags[word] = cfd[word_lower].max()
    else:
        likely_tags[word] = 'NN'  # Default to noun
    #print(f"Word: {word}, Likely Tag: {likely_tags[word]}")
lookup_tagger = nltk.UnigramTagger(model=likely_tags)


# Print the 20 most common words
'''
print("Most common words in Pokemon descriptions:")
for word, freq in fdist.most_common(20):
    print(f"{word}: {freq}")
'''


#Test the taggers
def test_tagger(tagger, tagged_descriptions):
    correct = 0
    total = 0
    for pokemon_id, tagged_description in tagged_descriptions.items():
        words = [word for word, _ in tagged_description]
        tags = [tag for _, tag in tagged_description]
        predicted_tags = tagger.tag(words)
        for (word, tag), (_, predicted_tag) in zip(tagged_description, predicted_tags):
            if tag == predicted_tag:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0


# N-gram tagger

training_data = [] 
#80 % of entries for training
for pokemon in pokemons[:int(len(pokemons) * 0.8)]:
    description = pokemon.get_info()
    tokens = nltk.word_tokenize(description)
    tagged_tokens = nltk.pos_tag(tokens)
    training_data.append(tagged_tokens)

testing_data = []
# Rest 20% for testing
for pokemon in pokemons[int(len(pokemons) * 0.8):]:
    description = pokemon.get_info()
    tokens = nltk.word_tokenize(description)
    tagged_tokens = nltk.pos_tag(tokens)
    testing_data.append(tagged_tokens)
    
# Unigram tagger
unigram_tagger = nltk.UnigramTagger(training_data, backoff=default_tagger)
# Bigram tagger
bigram_tagger = nltk.BigramTagger(training_data, backoff=unigram_tagger)
# Trigram tagger
trigram_tagger = nltk.TrigramTagger(training_data, backoff=bigram_tagger)

# Test the n-gram taggers, use testing data
def test_ngram_tagger(tagger, testing_data):
    correct = 0
    total = 0
    for tagged_tokens in testing_data:
        words = [word for word, _ in tagged_tokens]
        tags = [tag for _, tag in tagged_tokens]
        predicted_tags = tagger.tag(words)
        for (word, tag), (_, predicted_tag) in zip(tagged_tokens, predicted_tags):
            if tag == predicted_tag:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0

#Tagger combination
# Unigram tagger + lookup tagger
t_ul = nltk.UnigramTagger(training_data, backoff=lookup_tagger)


# Unigram tagger + regexp tagger
t_ur = nltk.UnigramTagger(training_data, backoff=regexp_tagger)


# Bigram tagger + lookup tagger
t_bl = nltk.BigramTagger(training_data, backoff=lookup_tagger)

# Bigram tagger + regexp tagger
t_br = nltk.BigramTagger(training_data, backoff=regexp_tagger)

# Trigram tagger + lookup tagger
t_tl = nltk.TrigramTagger(training_data, backoff=lookup_tagger)
# Trigram tagger + regexp tagger
t_tr = nltk.TrigramTagger(training_data, backoff=regexp_tagger)




#Evaluate the taggers performance : time eval
import time
def evaluate_tagger_performance(tagger, testing_data):
    start_time = time.time()
    accuracy = test_ngram_tagger(tagger, testing_data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return accuracy, elapsed_time
# Evaluate the performance of each tagger
taggers = {
    "Default": default_tagger,
    "Regexp": regexp_tagger,
    "Lookup": lookup_tagger,
    "Unigram": unigram_tagger,
    "Bigram": bigram_tagger,
    "Trigram": trigram_tagger,
    "Unigram + Lookup": t_ul,
    "Unigram + Regexp": t_ur,
    "Bigram + Lookup": t_bl,
    "Bigram + Regexp": t_br,
    "Trigram + Lookup": t_tl,
    "Trigram + Regexp": t_tr
}
performance_results = {}
for name, tagger in taggers.items():
    accuracy, elapsed_time = evaluate_tagger_performance(tagger, testing_data)
    performance_results[name] = {
        "Accuracy": accuracy,
        "Time (seconds)": elapsed_time
    }
# Print the performance results
print("\nTagger Performance Results:")
for name, results in performance_results.items():
    print(f"{name}: Accuracy = {results['Accuracy']:.2f}, Time = {results['Time (seconds)']:.4f}")
    
