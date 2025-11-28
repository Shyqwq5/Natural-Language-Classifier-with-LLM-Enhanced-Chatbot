from newsclassifier import NewsClassifier

news_classifier = NewsClassifier("trained_lg_model.pkl")

print('classifier running, enter `exit` to quit')
while True:
    news_headline = input('Enter a news headline to classify:')
    if news_headline == 'exit':
        break
    classification = news_classifier.classify_review(news_headline)

    print("Classification Report:",classification)
