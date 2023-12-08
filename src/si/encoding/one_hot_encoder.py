import numpy as np

class OneHotEncoder:
    '''
    One-hot encoder for sequences.
    '''
    def __init__(self, padder: str, max_length: int = None):
        '''
        Initialize the encoder.

        Paramenters
        ----------
        padder: str
            The character to use for padding.
        max_length: int
            The maximum length of the sequences.

        Attributes
        ----------
        alphabet: set
            The unique characters in the sequences.
        char_to_index: dict
            Dictionary mapping characters in the alphabet to indices.
        index_to_char: dict
            Dictionary mapping indices to characters in the alphabet (reverse of char_to_index).
        '''
        self.padder = padder
        self.max_length = max_length

        self.alphabet = set()
        self.char_to_index = {}
        self.index_to_char = {}
    
    def fit(self, data: list[str]):
        '''
        Fit the encoder to the data.

        Parameters
        ----------
        data: list of str
            List of sequences (strings) to learn from

        Returns
        -------
        OneHotEncoder
            The fitted encoder.
        '''
        # get the maximum length of the sequences
        if self.max_length is None:
            self.max_length = max([len(sequence) for sequence in data])

        # get the unique characters in the sequences
        all_seq = "".join(data)
        self.alphabet = np.unique(list(all_seq))

        for i, char in enumerate(self.alphabet):
            self.char_to_index[char] = i
            self.index_to_char[i] = char

        # add the padder to the alphabet if it is not already there
        if self.padder not in self.alphabet:
            self.alphabet = np.append(self.alphabet, self.padder)
            max_index = max(self.char_to_index.values())
            new_index = max_index + 1
            self.char_to_index[self.padder] = new_index
            self.index_to_char[new_index] = self.padder

        return self
    

    def transform(self, data: list[str]) -> np.ndarray:
        '''
        Transform the input data into a one-hot encoded matrix.

        Parameters
        ----------
        data: list of str
            List of sequences (strings) to encode.

        Returns
        -------
        numpy.ndarray
            The one-hot encoded matrix.
        '''

        #trim the sequences to max length
        data = [sequence[:self.max_length] for sequence in data]

        #pad the sequences to max length
        data = [sequence.ljust(self.max_length, self.padder) for sequence in data]

        #create the one-hot encoded matrix ()), i.e., for each sequence you wil end up with a matrix of shape max_length x alphabet_size)
        one_hot_encode = np.zeros((len(data), self.max_length, len(self.alphabet)))
        matrix_identity = np.eye(len(self.alphabet))

        # for each sequence, fill the corresponding matrix with the one-hot encoding of each character
        for i, sequence in enumerate(data):
            for j, char in enumerate(sequence):
                char_index = self.char_to_index[char]
                one_hot_encode[i, j] = matrix_identity[char_index]

        return np.array(one_hot_encode)
    

    def fit_transform(self, data: list[str]) -> np.ndarray:
        '''
        Fit the encoder to the data and transform the data into a one-hot encoded matrix.

        Parameters
        ----------
        data: list of str
            List of sequences (strings) to encode.

        Returns
        -------
        numpy.ndarray
            The one-hot encoded matrix.
        '''
        self.fit(data)
        return self.transform(data)
    

    def inverse_transform(self, data: np.ndarray) -> list[str]:
        '''
        Decode the one-hot encoded matrix into a list of the original sequences.

        Parameters
        ----------
        data: numpy.ndarray
            The one-hot encoded matrix.

        Returns
        -------
        list of str
            The list of the original sequences.
        '''

        # retrieve the index of the maximum value in each row of the one hot encoded matrix (for each original sequence)
        indexes_per_matrix = [np.argmax(one_hot_matrix, axis=1) for one_hot_matrix in data]

        # retrieve the character corresponding to each index
        total_sequences = [[self.index_to_char[idx] for idx in index] for index in indexes_per_matrix]

        # join the retrieved characters to form the original sequences
        original_seqs = ["".join(sequence) for sequence in total_sequences]

        # trim the sequences to the original length
        final_seqs = [sequence.rstrip(self.padder) for sequence in original_seqs]

        return final_seqs
    
    


    
if __name__ == "__main__":
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder


    # example data
    data = np.array(['abc', 'def', 'ab', 'de'])

    # encoder
    custom_encoder = OneHotEncoder(padder='_')
    custom_encoded = custom_encoder.fit_transform(data)
    custom_decoded = custom_encoder.inverse_transform(custom_encoded)

    # comparing results
    print("Categories:\n", custom_encoder.alphabet)
    print("Custom Encoder Encoded:\n", custom_encoded)
    print("Custom Encoder Decoded:\n", custom_decoded)



