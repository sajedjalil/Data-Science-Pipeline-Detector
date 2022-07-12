import pandas as pd
import tqdm
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=20)

np = pd.np

train_path = '../input/train.csv'
test_path = '../input/test.csv'

try:
   input = raw_input
except NameError:
   pass

try:
    basestring
except:
    basestring = str

try:
    unicode
except:
    unicode = lambda x: x


def get_label_matrix(df):
    return df.iloc[:, 2:].__array__()


def to_label(preds, class_names):
    predicted_labels = []
    for pred in preds:
        labels = []
        for index, binary in enumerate(pred):
            if binary == 1:
                labels.append(class_names[index])
        predicted_labels.append(labels)
    return predicted_labels


def train_deep_bow_model(df, class_names, train=True):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPClassifier
    from sklearn.externals import joblib
    from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer

    if train:
        vec = TfidfVectorizer(binary=False, min_df=10, ngram_range=(1, 5),
            analyzer='char')
        # clf = OneVsRestClassifier(LogisticRegression())
        clf = MLPClassifier(
            hidden_layer_sizes=(600, 300),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=200,
            early_stopping=True)

        # clustering
        # TODO: try LDA as well
        svd = TruncatedSVD(1000)
        normalizer = Normalizer(copy=False)
        pipe = make_pipeline(vec, svd, normalizer)

        X = pipe.fit_transform(df.comment_text)
        y = get_label_matrix(df)
 
        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))

        clf.fit(X, y)

        # joblib.dump(pipe, 'deep_pipe.pkl')
        # joblib.dump(clf, 'deep_clf.pkl')

    else:
        # pipe = joblib.load('deep_pipe.pkl')
        # clf = joblib.load('deep_clf.pkl')
        pass

    def predict(text, use_prob=False):
        if isinstance(text, basestring):
            text = [text]
        text = [unicode(t) for t in text]
        X = pipe.transform(text)
        if use_prob:
            probs = clf.predict_proba(X)
            return probs
        else:
            preds = clf.predict(X)
            labels = to_label(preds, class_names)
            return labels

    return predict


def write_submission_file(df, probs, class_names, path='submission.csv'):
    print_df = \
        pd.concat([df.id, pd.DataFrame(probs[:, :6])],
            axis=1, ignore_index=True)
    print_df.columns = ['id'] + class_names[:6]
    print_df.to_csv(path, index=False)


def main():
    train_df = pd.read_csv(train_path, encoding='utf8')
    test_df = pd.read_csv(test_path, encoding='utf8')

    train_df.fillna('__missing__', inplace=True)
    test_df.fillna('__missing__', inplace=True)

    train_df['no_label'] = (train_df.iloc[:, 2:].sum(axis=1) == 0).astype(int)

    class_names = train_df.columns[2:].tolist()

    print(train_df.head())

    predict = train_deep_bow_model(train_df, class_names)

    print('Testing and writing submission file')
    test_probs = predict(test_df.comment_text, use_prob=True)
    write_submission_file(test_df, test_probs, class_names,
        path='submission.csv')


if __name__ == '__main__':
    main()