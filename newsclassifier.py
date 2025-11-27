import pickle


class NewsClassifier:
    def __init__(self,model_path,mapping_path = 'mapping.pkl',vectorizer_path = 'vectorizer.pkl'):
        self.model = self.load_from_path(model_path)
        self.vectorizer = self.load_from_path(vectorizer_path)
        self.mapping = self.load_mapping(mapping_path)

    def load_from_path(self,path):
        with open(path, 'rb') as f:
            result = pickle.load(f)
        return result

    def load_mapping(self,path):
        mapping = self.load_from_path(path)
        reversed_mapping = {}
        for key,value in mapping.items():
            reversed_mapping[value] = key
        return reversed_mapping

    def classify_review(self,input):
        vector_input = self.vectorizer.transform([input])
        pred = self.model.predict(vector_input)
        return self.mapping[int(pred[0])]




