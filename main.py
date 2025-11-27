from newsclassifier import NewsClassifier

news_classifier = NewsClassifier("trained_lg_model.pkl")

classification = news_classifier.classify_review("NHS trust fined £200k over vulnerable girl's death")

print(classification)