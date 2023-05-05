import numpy as np

class DataReader:
    def __init__(self, path, seq_length):
        with open(path, "r") as f:
            self.data = f.read()

        self.seq_length = seq_length
        self.pointer = 0

        # Create a list of all unique characters in the data
        self.chars = sorted(list(set(self.data)))

        # Create a dictionary that maps each unique character to an integer index
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # Calculate the total number of characters in the data and the size of the vocabulary
        self.data_size = len(self.data)
        self.vocab_size = len(self.chars)

    def next_batch(self):
        # Determine the start and end indices of the input and target sequences
        input_start = self.pointer
        input_end = self.pointer + self.seq_length
        target_start = input_start + 1
        target_end = input_end + 1

        # Convert the characters in the input and target sequences to integer indices using the char_to_ix dictionary
        inputs = [self.char_to_ix[ch] for ch in self.data[input_start:input_end]]
        targets = [self.char_to_ix[ch] for ch in self.data[target_start:target_end]]

        # Update the pointer to point to the start of the next input sequence
        self.pointer += self.seq_length

        # If the pointer has reached the end of the data, reset it to the start of the data
        if self.pointer + self.seq_length >= self.data_size:
            self.pointer = 0

        return inputs, targets

    def just_started(self):
        return self.pointer == 0

