import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest

# Load model
nlp = spacy.load('en_core_web_sm')

# Input text
text = ("Paris is the capital city of France. "
        "It is famous for the Eiffel Tower and delicious food. "
        "Paris is known for its art, fashion, and culture. "
        "Millions of tourists visit Paris every year.")

# Process the text
doc = nlp(text)

# Keep "is" and "of"
stopwords = list(STOP_WORDS)
for w in ["is", "of"]:
    if w in stopwords:
        stopwords.remove(w)

# Extract keywords
keywords = []
pos_tags = ['PROPN', 'ADJ', 'NOUN', 'VERB']

for token in doc:
    if token.text.lower() in stopwords or token.text in punctuation:
        continue
    if token.pos_ in pos_tags:
        keywords.append(token.text)

# Keyword frequency
freq = Counter(keywords)
print("Top 5 Keywords:", freq.most_common(5))

# Sentence strength
sent_strength = {}
for sent in doc.sents:
    for word in sent:
        if word.text in freq:
            sent_strength[sent] = sent_strength.get(sent, 0) + freq[word.text]


summary = nlargest(2, sent_strength, key=sent_strength.get)
print("\nSummary:")
for sent in summary:
    print("-", sent.text)
