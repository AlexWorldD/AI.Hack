from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.preprocessing import LabelEncoder


class EncodeCategorical(BaseEstimator, TransformerMixin):
    """
    Encodes a specified list of columns or all columns if None.
    """

    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = None

    def fit(self, data, target=None):
        """
        Expects a data frame with named columns to encode.
        """
        # Encode all columns if columns is None
        if self.columns is None:
            self.columns = data.columns

        # Fit a label encoder for each column in the data frame
        self.encoders = {
            column: LabelEncoder().fit(data[column])
            for column in self.columns
        }
        return self

    def export(self):
        """
        Uses for export encoders.
        """
        np.save('LabelEncoding/Columns.npy', self.columns)
        for name in self.columns:
            np.save('LabelEncoding/' + name + '.npy', self.encoders[name].classes_)

    def set(self):
        """
        Uses for import encoders for prediction set.
        """
        self.columns = np.load('LabelEncoding/Columns.npy')
        self.encoders = {
            column: LabelEncoder()
            for column in self.columns
        }
        for name in self.columns:
            self.encoders[name].classes_ = np.load('LabelEncoding/' + name + '.npy')

    def transform(self, data):
        """
        Uses the encoders to transform a data frame.
        """
        output = data.copy()
        for column, encoder in self.encoders.items():
            output[column] = encoder.transform(data[column])

        return output

    def fit_transform(self, data, target=None):
        return self.fit(data, target).transform(data)
