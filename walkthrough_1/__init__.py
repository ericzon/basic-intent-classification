# Install the necessary packages: Make sure you have NLTK installed in your Python environment. You can install it using pip: pip install nltk.

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def start():
    # Create the corpus: Define a list of labeled sentences representing different intents.
    corpus = [
        ("I am hungry?", "food"),
        ("I would like to eat something?", "food"),
        ("do you have any hamburguer?", "food"),
        ("I'll eat a salad", "food"),
        ("I'm starving", "food"),
        ("I will have meal soon", "food"),
        ("I'll take something for eating", "food"),

        ("I'm thirsty?", "drink"),
        ("I would like to drink something?", "drink"),
        ("do you have some water?", "drink"),
        ("I'll drink a glass of coke", "drink"),
        ("I drank a bottle of lemonade", "drink"),
        ("we were drinking until late", "drink"),
    ]

    # Preprocess the corpus: Preprocess the corpus by tokenizing, lemmatizing, and removing stopwords.
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    preprocessed_corpus = []
    for sentence, intent in corpus:
        tokens = word_tokenize(sentence.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if token not in stop_words]
        preprocessed_corpus.append((' '.join(tokens), intent))

    # Split the corpus into training and testing sets: Split the preprocessed corpus into training and testing sets.
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    train_corpus, test_corpus = train_test_split(preprocessed_corpus, test_size=0.2, random_state=42)

    # Train the intent classifier: Create a pipeline that includes a TF-IDF vectorizer and a Support Vector Machine (SVM) classifier. Fit the pipeline on the training corpus.
    pipeline = Pipeline([
        # as a feature representation technique
        ('tfidf', TfidfVectorizer()),
        # as a classification engine
        ('svm', SVC(kernel='linear'))
    ])

    X_train, y_train = zip(*train_corpus)
    pipeline.fit(X_train, y_train)

    # Evaluate the intent classifier: Use the trained classifier to predict the intents for the test corpus and evaluate its performance.
    X_test, y_test = zip(*test_corpus)
    accuracy = pipeline.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")

    # Use the intent classifier: Once you have trained and evaluated the classifier, you can use it to predict the intent of new sentences.

    sentence = "I would like to eat a hamburguer"
    predicted_intent = pipeline.predict([sentence])
    print(f"sentence: {sentence}. intent classification: {predicted_intent[0]}")

    sentence = "I'll be drinking a glass of wine"
    predicted_intent = pipeline.predict([sentence])
    print(f"sentence: {sentence}. intent classification: {predicted_intent[0]}")

    sentence = "I want to eat something"
    predicted_intent = pipeline.predict([sentence])
    print(f"sentence: {sentence}. intent classification: {predicted_intent[0]}")
