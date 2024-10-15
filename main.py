import pandas as pd
import warnings
import snowballstemmer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn import naive_bayes

warnings.simplefilter(action="ignore", category=FutureWarning)

df = pd.read_csv(
    "yorumsepeti.csv", delimiter=";", encoding="utf-8-sig", on_bad_lines="skip"
)

df.columns = df.columns.str.strip()

df["review"] = df["review"].fillna("Bu alanda bir yorum yok")

df = df.sample(n=10000, random_state=20)

stop_words = set(stopwords.words("turkish"))
stemmer = snowballstemmer.stemmer("turkish")


def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char.isalpha() or char.isspace()])
    words = text.split()
    words = [stemmer.stemWord(word) for word in words if word not in stop_words]
    return " ".join(words)


df["Kök Metin"] = df["review"].apply(preprocess_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
Xtf = vectorizer.fit_transform(df["Kök Metin"]).astype("float32")

train_x, test_x, train_y, test_y = model_selection.train_test_split(
    Xtf, df["review"], test_size=0.30, random_state=42
)

nb_multi = naive_bayes.MultinomialNB()
nb_model_multi = nb_multi.fit(train_x, train_y)
y_pred_nb_multi = nb_model_multi.predict(test_x)
print("Multinomial Accuracy:", accuracy_score(test_y, y_pred_nb_multi, normalize=True))
print(classification_report(test_y, y_pred_nb_multi))


def predict_sentiment(new_review):

    processed_review = preprocess_text(new_review)

    tfidf_review = vectorizer.transform([processed_review])

    prediction = nb_model_multi.predict(tfidf_review)
    return prediction[0]


new_comment = input("Tahmin edilmesini istediğiniz yorumu girin: ")
predicted_label = predict_sentiment(new_comment)
print(f"Girilen yorumun tahmin edilen duygu sınıfı: {predicted_label}")
