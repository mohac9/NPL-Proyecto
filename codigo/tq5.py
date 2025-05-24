import nltk
import parser as ps

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
for pokemon_id, tagged_description in tagged_descriptions.items():
    print(f"ID: {pokemon_id}, Tagged Description: {tagged_description}")

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
#Get most common words 
from nltk.probability import FreqDist
most_common_words = nltk.FreqDist(
    word for tagged_description in tagged_descriptions.values() 
    for word, _ in tagged_description
).most_common(100)

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

# Test the default tagger
default_accuracy = test_tagger(default_tagger, tagged_descriptions)
print(f"Default Tagger Accuracy: {default_accuracy:.2f}")

# Test the regular expression tagger
regexp_accuracy = test_tagger(regexp_tagger, tagged_descriptions)
print(f"Regexp Tagger Accuracy: {regexp_accuracy:.2f}")

#Test performance 
def performance(cfd, wordlist):
    lt = dict((word, cfd[word].max()) for word in wordlist)
    baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))

def display():
    import pylab
    word_freqs = nltk.FreqDist(brown.words(categories='news')).most_common()
    words_by_freq = [w for (w, _) in word_freqs]
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    sizes = 2 ** pylab.arange(15)
    perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()

# Display the performance graph
display()