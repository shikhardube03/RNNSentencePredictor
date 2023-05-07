from DataLoader import DataReader 
from RecurrantNerualNetwork import RNN 


#read text from the "input.txt" file
data_reader = DataReader("input.txt", 21)
trainedRNN = RNN(hidden_size=100, vocab_size=data_reader.vocab_size,seq_length=21,learning_rate=1e-1)
trainedRNN.train(data_reader)


trainedRNN.predict(data_reader, 'basic', 1000)
# trainedRNN.plot_loss(trainedRNN.losses)
# trainedRNN.predict(data_reader, 'image', 50)