# Hybrid-Machine-Translation-Combining-SMT-and-Neural-Word-Embeddings

### Description:
This repository presents a hybrid approach to machine translation, combining traditional Statistical Machine Translation (SMT) methods with neural network-based word embeddings using Word2Vec and Marian Neural Machine Translation (NMT). The system leverages the strengths of both approaches: SMT for handling phrase-based translations and word embeddings for enhancing semantic understanding, followed by an NMT model for final translation refinement.

### Features:
- **Word Embedding Training**: Train custom word embeddings using the Word2Vec model on large corpora (e.g., Brown corpus).
- **SMT Output Enhancement**: Use word embeddings to enhance the quality of SMT-generated translations by improving semantic relevance and fluency.
- **Neural Machine Translation**: Perform final translation refinement using MarianMT, a state-of-the-art pre-trained NMT model.
- **Multi-language Support**: Supports translation between English and multiple target languages, including Hindi (`en-hi`) and Spanish (`en-es`).
- **Pipeline Integration**: A complete machine translation pipeline that integrates SMT, word embeddings, and NMT in a seamless flow.

### Requirements:
- Python 3.x
- Gensim
- NLTK
- Transformers (Hugging Face)
- PyTorch

### How to Use:
1. **Train Word Embeddings**:
   - Train the Word2Vec model on any corpus of your choice.
   - Pre-trained embeddings for word similarity and enhancement in translation tasks.
   
2. **Generate SMT Output**:
   - Use an SMT model to generate baseline translations (or input a pre-generated SMT output).

3. **Enhance SMT Output**:
   - Improve the SMT output using word embeddings to refine semantic accuracy and fluency.

4. **Translate Using NMT**:
   - Pass the enhanced SMT output through the MarianMT model to generate the final translated text.

### Example Usage:
```bash
# Run the full translation pipeline
python translate.py --input "The government is planning new policies."
```

### Contributions:
Contributions are welcome! Please feel free to submit pull requests or report issues to help improve the project.
