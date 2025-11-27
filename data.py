import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def get_data(path='cleaned_news_data.csv'):
    df = pd.read_csv(path)
    mapping = {}
    index = 0
    for topic in df["topic"].unique():
        mapping[topic] = index
        index +=1

    y = df["topic"].map(mapping).values
    X = df['title'].values
    return X,y,mapping

# path='cleaned_news_data.csv'
# df = pd.read_csv(path)
# pd.set_option('display.max_colwidth', None)
# print(df.head())

def get_vectorized_data(path='cleaned_news_data.csv',test_size = 0.3,RANDOM_SEED = 28):
    X, y,mapping = get_data(path)

    RANDOM_SEED = 28

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_SEED
    )

    vectorizer = TfidfVectorizer(
        max_features=40000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)
    return X_train,X_test,y_train, y_test,mapping,vectorizer