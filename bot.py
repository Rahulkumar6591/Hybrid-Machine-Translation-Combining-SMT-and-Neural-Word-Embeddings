# Import necessary libraries
from gensim.models import Word2Vec
from nltk.corpus import brown
import nltk
from transformers import MarianMTModel, MarianTokenizer

# Download the necessary NLTK data
nltk.download('brown')
nltk.download('punkt')

# 1. Word Embedding Training using Word2Vec
# -----------------------------------------
# Load the Brown corpus from NLTK (you can replace this with your own corpus)
sentences = brown.sents()

# Train a Word2Vec model on the corpus
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)

# Save the trained Word2Vec model
word2vec_model.save("word2vec_brown.model")

# To load the saved Word2Vec model:
word2vec_model = Word2Vec.load("word2vec_brown.model")

# Example: Get the vector representation of a word
word_vector = word2vec_model.wv['government']
print(f"Embedding vector for 'government': {word_vector}")

# Example: Find words similar to 'government'
similar_words = word2vec_model.wv.most_similar('government')
print(f"Words similar to 'government': {similar_words}")

# 2. Pre-trained Neural Machine Translation (NMT) using MarianMT
# -------------------------------------------------------------
# MarianMT is a pre-trained NMT model available in Hugging Face. We will use the model for English to Spanish translation.

# Load the MarianMT model and tokenizer for English to Spanish translation
tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-hi')
nmt_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-hi')

# Function to translate text using MarianMT
def translate_text(text, src_lang="en", tgt_lang="es"):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Generate translation
    translated_tokens = nmt_model.generate(**inputs)
    
    # Decode the translated sentence
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# Example translation from English to Spanish
english_text = "The government is introducing new policies."
translated_text = translate_text(english_text)
print(f"Original: {english_text}")
print(f"Translated: {translated_text}")

# 3. Hybrid Approach: Combining Word Embeddings with Translation
# --------------------------------------------------------------
# We can now combine the Word2Vec embeddings with the translation model. Let's assume you have SMT output,
# and we want to use word embeddings to improve the translation. We will use the Word2Vec embeddings 
# to look for semantically similar words and enhance the SMT step conceptually.

# This function finds the top-3 similar words based on the word embeddings
def get_similar_words(word, topn=3):
    if word in word2vec_model.wv.key_to_index:  # Check if the word exists in the Word2Vec vocabulary
        return word2vec_model.wv.most_similar(word, topn=topn)
    return []

# Example usage of enhancing translation using word embeddings:
def enhance_translation_with_embeddings(text):
    words = text.split()
    enhanced_words = []
    
    # Iterate over each word and try to enhance it with word embeddings
    for word in words:
        similar_words = get_similar_words(word)
        if similar_words:
            # Replace the word with its top similar word for demonstration (or use SMT output here)
            enhanced_words.append(similar_words[0][0])  # Use the most similar word
        else:
            enhanced_words.append(word)  # If no similar words, use the original word

    # Join the enhanced words back into a sentence
    enhanced_sentence = ' '.join(enhanced_words)
    return enhanced_sentence

# Example: Enhance an English sentence with word embeddings
smt_output = "The government is planning new policies."  # Assume this is SMT output
enhanced_smt_output = enhance_translation_with_embeddings(smt_output)
print(f"Original SMT Output: {smt_output}")
print(f"Enhanced SMT Output: {enhanced_smt_output}")

# Now, translate the enhanced SMT output using the NMT model
final_translation = translate_text(enhanced_smt_output)
print(f"Final Translation after Enhancement: {final_translation}")

# 4. Putting It All Together: A Full Pipeline
# -------------------------------------------
def machine_translation_pipeline(text):
    # Step 1: Assume we have SMT output (can be replaced with real SMT step)
    smt_output = text  # Placeholder for actual SMT output
    
    # Step 2: Enhance SMT output with word embeddings
    enhanced_output = enhance_translation_with_embeddings(smt_output)
    
    # Step 3: Translate the enhanced output using the NMT model
    final_translation = translate_text(enhanced_output)
    
    return final_translation

# Example: Full pipeline for machine translation
input_sentence = input("Text: ")
output_translation = machine_translation_pipeline(input_sentence)
print(f"Final Translation: {output_translation}")
