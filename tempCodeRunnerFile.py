   with open('model.pkl', 'wb') as file: 
        pickle.dump(classifier, file)

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)