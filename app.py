import os
import pickle
from flask import Flask, render_template, request
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Cleaning data:
stop_words = set(stopwords.words('english'))
new_stop_words = ["fig", "figure", "image", "sample", "using",
                  "show", "result", "large",
                  "also", "one", "two", "three",
                  "four", "five", "seven", "eight", "nine"]
stop_words = list(stop_words.union(new_stop_words))

def preprocess_text(txt):
    # Lower case
    txt = txt.lower()
    # Remove HTML tags
    txt = re.sub(r"<.*?>", " ", txt)
    # Remove special characters and digits
    txt = re.sub(r"[^a-zA-Z]", " ", txt)
    # tokenization
    txt = nltk.word_tokenize(txt)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    txt = [word for word in txt if word not in stop_words]
    # Remove words less than three letters
    txt = [word for word in txt if len(word) >= 3]
    # Lemmatize
    lmtr = WordNetLemmatizer()
    txt = [lmtr.lemmatize(word) for word in txt]

    return " ".join(txt)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    document = request.files['file']
    if document.filename == '':
        return render_template('index.html', error='No document selected')

    if document:
        text = document.read().decode('utf-8', errors='ignore')
        cleaned_text = preprocess_text(text)

        # Initialize TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer()

        # Fit and transform the preprocessed text
        tfidf_matrix = tfidf_vectorizer.fit_transform([cleaned_text])

        # Get feature names
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # Get TF-IDF vector for the cleaned text
        tfidf_vector = tfidf_matrix.toarray()[0]

        # Combine feature names with their corresponding TF-IDF values
        keywords = dict(zip(feature_names, tfidf_vector))

        return render_template('keywords.html', keywords=keywords)

    return render_template('index.html')

@app.route('/search_keywords', methods=['POST'])
def search_keywords():
    search_query = request.form['search']
    if search_query:
        keywords = []
        for keyword in feature_names:
            if search_query.lower() in keyword.lower():
                keywords.append(keyword)
                if len(keywords) == 20:  # Limit to 20 keywords
                    break
        return render_template('keywordslist.html', keywords=keywords)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
