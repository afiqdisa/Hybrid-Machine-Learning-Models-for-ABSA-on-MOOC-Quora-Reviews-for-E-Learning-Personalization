import nltk

# The specific tagger that is missing
nltk.download('averaged_perceptron_tagger') 

# Other common packages that your preprocess_text function will likely need
nltk.download('punkt')        # Used for tokenizing sentences into words
nltk.download('wordnet')      # Used for lemmatization
nltk.download('stopwords')    # Used for removing stop words

print("--- NLTK data downloaded successfully. You can now run your Flask app. ---")