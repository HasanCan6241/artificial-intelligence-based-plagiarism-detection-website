import re
import nltk
import pickle
from django.shortcuts import render
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
from sklearn.exceptions import ConvergenceWarning
import os

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

def anasayfa(request):
    return render(request, 'anasayfa.html')



with open('logistic_regression_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    loaded_vectorizer = pickle.load(file)


def intihal_tespit(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        stop_words = set(stopwords.words('english'))

        words = [word for word in text.split() if word not in stop_words]

        lemmatizer = WordNetLemmatizer()

        words_lemmatized = [lemmatizer.lemmatize(word) for word in words]

        processed_text = ' '.join(words_lemmatized)


        X_new = loaded_vectorizer.transform([processed_text])
        predictions = int(loaded_model.predict(X_new))

        return render(request, 'intihal_tespit.html', {'text': text, 'predictions': predictions})

    return render(request, 'intihal_tespit.html')

