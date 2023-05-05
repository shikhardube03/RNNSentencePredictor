import numpy as np

class DataReader:
    def __init__(self, path, seq_length):
        #uncomment below , if you dont want to use any file for text reading and comment next 2 lines
        # self.data = "some really long text to test this. maybe not perfect but should get you going."
        self.fp = open(path, "r")
        self.data = self.fp.read()
        #find unique chars
        chars = list(set(self.data))
        #create dictionary mapping for each char
        self.char_to_ix = {ch:i for (i,ch) in enumerate(chars)}
        self.ix_to_char = {i:ch for (i,ch) in enumerate(chars)}
        #total data
        self.data_size = len(self.data)
        #num of unique chars
        self.vocab_size = len(chars)
        self.pointer = 0
        self.seq_length = seq_length

    def next_batch(self):
        input_start = self.pointer
        input_end = self.pointer + self.seq_length
        inputs = [self.char_to_ix[ch] for ch in self.data[input_start:input_end]]
        targets = [self.char_to_ix[ch] for ch in self.data[input_start+1:input_end+1]]
        self.pointer += self.seq_length
        if self.pointer + self.seq_length + 1 >= self.data_size:
            # reset pointer
            self.pointer = 0
        return inputs, targets

    def just_started(self):
        return self.pointer == 0

    def close(self):
        self.fp.close()    

# seq_length = 25
# #read text from the "input.txt" file
# data_reader = DataReader("input.txt", seq_length)
# rnn = RNN(hidden_size=100, vocab_size=data_reader.vocab_size,seq_length=seq_length,learning_rate=1e-1)
# rnn.train(data_reader)

# rnn.predict(data_reader, 'get', 50)
# rnn.predict(data_reader, 'image', 50)
