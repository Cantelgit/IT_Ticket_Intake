# Company: Cantel Medical
# Organization: IT
# Department: Advanced Analytics
# Author: Garrett Eichhorn

"""

This python script is intended to build a Machine Learning Model to make predictions. Specifically, I'm
employing a Supervised Classification model using Natural Language Processing (NLP) to make predictions for incoming
ServiceNow (SNOW) tickets. The model learns from text fields using Logistic Regression, mapping important features
to defined categories: Portfolio and Assignment Group.

This script will read an excel file of tickets (keeping only the relevant columns), pre-process the text for predictions,
split the data into training / test sets, instantiate a pipeline, and output the model as a Pickle file.

"""

########################################################################################################################

# Read the data and perform basic pre-processing of the dataframe

import pandas as pd
import numpy as np
from warnings import warn

# Read the excel file (downloaded initially from MongoDB) which contains 27,000 records
full_file = pd.read_excel("C:\\Users\garrett.eichhorn\PycharmProjects\MachineLearning\Gather Input Data\\final_storage.xlsx", index_col='number')

# Keep relevant columns and convert type(), drop records without feature: portfolio
text_columns = full_file[['u_portfolio', 'description', 'short_description']]
text_columns['portfolio'] = text_columns['u_portfolio'].astype('category')
cat_df = text_columns.dropna()

# Create dummy labels for model processing
dummy_labels = pd.get_dummies(cat_df[['portfolio']], prefix_sep='_')

# Create text only data for model processing
text_only_data = cat_df.drop(columns = ['u_portfolio', 'portfolio'])

# Keep a "full" dataframe for quick analysis of both labels and text
thicc_dataframe = pd.concat([cat_df, dummy_labels], axis=1)

# -----------------------------------------------------------------------------------------------------------------------
# Multi-label functions imported from https://github.com/drivendataorg/box-plots-sklearn/blob/master/src/data/multilabel.py

def multilabel_sample(y, size=1000, min_count=5, seed=None):
    """ Takes a matrix of binary labels `y` and returns
        the indices for a sample of size `size` if
        `size` > 1 or `size` * len(y) if size =< 1.
        The sample is guaranteed to have > `min_count` of
        each label.
    """
    try:
        if (np.unique(y).astype(int) != np.array([0, 1])).any():
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError('multilabel_sample only works with binary indicator matrices')

    if (y.sum(axis=0) < min_count).any():
        raise ValueError('Some classes do not have enough examples. Change min_count if necessary.')

    if size <= 1:
        size = np.floor(y.shape[0] * size)

    if y.shape[1] * min_count > size:
        msg = "Size less than number of columns * min_count, returning {} items instead of {}."
        warn(msg.format(y.shape[1] * min_count, size))
        size = y.shape[1] * min_count

    rng = np.random.RandomState(seed if seed is not None else np.random.randint(1))

    if isinstance(y, pd.DataFrame):
        choices = y.index
        y = y.values
    else:
        choices = np.arange(y.shape[0])

    sample_idxs = np.array([], dtype=choices.dtype)

    # first, guarantee > min_count of each label
    for j in range(y.shape[1]):
        label_choices = choices[y[:, j] == 1]
        label_idxs_sampled = rng.choice(label_choices, size=min_count, replace=False)
        sample_idxs = np.concatenate([label_idxs_sampled, sample_idxs])

    sample_idxs = np.unique(sample_idxs)

    # now that we have at least min_count of each, we can just random sample
    sample_count = int(size - sample_idxs.shape[0])

    # get sample_count indices from remaining choices
    remaining_choices = np.setdiff1d(choices, sample_idxs)
    remaining_sampled = rng.choice(remaining_choices,
                                   size=sample_count,
                                   replace=False)

    return np.concatenate([sample_idxs, remaining_sampled])


def multilabel_sample_dataframe(df, labels, size, min_count=5, seed=None):
    """ Takes a dataframe `df` and returns a sample of size `size` where all
        classes in the binary matrix `labels` are represented at
        least `min_count` times.
    """
    idxs = multilabel_sample(labels, size=size, min_count=min_count, seed=seed)
    return df.loc[idxs]


def multilabel_train_test_split(X, Y, size, min_count=5, seed=None):
    """ Takes a features matrix `X` and a label matrix `Y` and
        returns (X_train, X_test, Y_train, Y_test) where all
        classes in Y are represented at least `min_count` times.
    """
    index = Y.index if isinstance(Y, pd.DataFrame) else np.arange(Y.shape[0])

    test_set_idxs = multilabel_sample(Y, size=size, min_count=min_count, seed=seed)
    train_set_idxs = np.setdiff1d(index, test_set_idxs)

    test_set_mask = index.isin(test_set_idxs)
    train_set_mask = ~test_set_mask

    return (X[train_set_mask], X[test_set_mask], Y[train_set_mask], Y[test_set_mask])

# -----------------------------------------------------------------------------------------------------------------------
# Sparse Interactions imported from https://github.com/drivendataorg/box-plots-sklearn/blob/master/src/features/SparseInteractions.py

from itertools import combinations

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


class SparseInteractions(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2, feature_name_separator="_"):
        self.degree = degree
        self.feature_name_separator = feature_name_separator

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not sparse.isspmatrix_csc(X):
            X = sparse.csc_matrix(X)

        if hasattr(X, "columns"):
            self.orig_col_names = X.columns
        else:
            self.orig_col_names = np.array([str(i) for i in range(X.shape[1])])

        spi = self._create_sparse_interactions(X)
        return spi

    def get_feature_names(self):
        return self.feature_names

    def _create_sparse_interactions(self, X):
        out_mat = []
        self.feature_names = self.orig_col_names.tolist()

        for sub_degree in range(2, self.degree + 1):
            for col_ixs in combinations(range(X.shape[1]), sub_degree):
                # add name for new column
                name = self.feature_name_separator.join(self.orig_col_names[list(col_ixs)])
                self.feature_names.append(name)

                # get column multiplications value
                out = X[:, col_ixs[0]]
                for j in col_ixs[1:]:
                    out = out.multiply(X[:, j])

                out_mat.append(out)

        return sparse.hstack([X] + out_mat)

# -----------------------------------------------------------------------------------------------------------------------
# Basic Pre-processing

import nltk
from nltk.corpus import stopwords
import re

# Function to pre-process the text vector. Convert to lower, remove stop words, punctuation, token/lemma.
def process_text(text_vector):

    # Convert the Series Object into a dataframe
    dataframe = pd.DataFrame({'sentences': text_vector})

    # Convert to lower case
    dataframe["sentences"] = dataframe["sentences"].str.lower()

    # Remove stopwords
    stop_words = stopwords.words('english')
    dataframe['sentences'] = dataframe['sentences'].apply(
        lambda row: ' '.join([word for word in row.split() if word not in (stop_words)]))

    # Remove punctuation
    dataframe["sentences"] = dataframe["sentences"].apply(lambda row: re.sub(r'[^\w\s]', '', row))

    # Tokenize the sentences for each row
    dataframe["tokenized_text"] = dataframe.apply(lambda row: nltk.word_tokenize(row["sentences"]), axis=1)

    # Lemmatize
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def lemmatize(s):
        s = [lemmatizer.lemmatize(word) for word in s]
        return s

    dataframe["clean_text"] = dataframe["tokenized_text"].apply(lambda row: lemmatize(row))

    # Count number of words in each sentence
    def word_count(row):
        initializer = 0
        for i in row:
            initializer += 1

        return initializer

    dataframe["word_count"] = dataframe["clean_text"].apply(lambda row: word_count(row))

    # Return full dataframe for quick examination

    return dataframe['sentences']

# -----------------------------------------------------------------------------------------------------------------------
# Import necessary ML modules, perform basic pre-processing, set-up the pipeline and build the basic model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib

TOKENS_ALPHANUMERIC = '[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+'

# Function to convert all text in each row of the dataframe to fit a single vector
def combine_text_columns(data_frame):

    # Drop non-text columns that are in the df
    text_data = data_frame

    # Replace nans with blanks
    text_data.fillna("", inplace=True)

    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)

text_vector = combine_text_columns(text_only_data)

final_text = process_text(text_vector)

#Split the data
X_train, X_test, y_train, y_test = train_test_split(final_text, dummy_labels, random_state=22)

print(X_test, y_test)

# Instantiate Pipeline object
pl = Pipeline([


    #('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)),
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2),
                                   min_df = 2,
                                   max_df = .95)),
    ('scalar', StandardScaler(with_mean=False)),
    ('clf', OneVsRestClassifier(LogisticRegression()))

])

model = pl.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

accuracy = model.score(X_test, y_test)
print("\nAccuracy on sample data - all data: ", accuracy)

joblib.dump(model, "LogisticRegression_model.pkl")