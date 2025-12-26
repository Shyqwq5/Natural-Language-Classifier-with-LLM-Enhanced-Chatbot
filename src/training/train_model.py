from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from src.training.data import get_vectorized_data
import pickle
from src.utils.get_path import get_path
import os
from src.data_processing.ingest import load_data,clean_and_save_news_data

ROOT = get_path(2)
cleaned_data_path = ROOT/"data"/"processed"/"cleaned_news_data.csv"
save_model_path = ROOT/"src"/'saved_model'
os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
os.makedirs(save_model_path, exist_ok=True)

if not os.path.isfile(cleaned_data_path):
    raw_data_path = ROOT/"data"/"raw"/"labeled_newscatcher_dataset.csv"
    url = "https://raw.githubusercontent.com/kotartemiy/topic-labeled-news-dataset/refs/heads/master/labeled_newscatcher_dataset.csv"
    df = load_data(raw_data_path,url)
    clean_and_save_news_data(df,cleaned_data_path)

RANDOM_SEED = 28
X_train,X_test,y_train, y_test,mapping,vectorizer = get_vectorized_data(cleaned_data_path,test_size = 0.3,RANDOM_SEED=RANDOM_SEED)



with open(save_model_path/'mapping.pkl', 'wb') as f:
    pickle.dump(mapping, f)


with open(save_model_path/'vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

#logisticregression model
lg = LogisticRegression(
    max_iter=3000,
    C=2.0,
    n_jobs=-1
)

lg.fit(X_train, y_train)
lg_pred = lg.predict(X_test)
with open(save_model_path/'trained_lg_model.pkl', 'wb') as f:
    pickle.dump(lg, f)
print("Accuracy:", accuracy_score(y_test, lg_pred))
print(classification_report(y_test, lg_pred))

#naive_bayes model
nb = MultinomialNB(alpha=0.1)
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
with open(save_model_path/'trained_nb_model.pkl', 'wb') as f:
    pickle.dump(nb, f)
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))


#randomfoest model
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=RANDOM_SEED
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
with open(save_model_path/'trained_rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("rf_accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
