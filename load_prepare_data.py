from tensorflow import keras

#Load IMDB dataset
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data()

lens = [len(train_data[i]) for i in range(len(train_data))]
max_length = max(lens)
min_length = min(lens)

word2index = keras.datasets.imdb.get_word_index()

word2index = dict([(key, value + 3) for (key, value) in word2index.items()])
word2index['<PAD>'] = 0
word2index['<START>'] = 1
word2index['<UNKNOWN>'] = 2
word2index['<UNUSED>'] = 3

index2word = dict([(value, key) for (key, value) in word2index.items()])

#Function to convert a sequence of int to a sequance of words
def getText(int_seq):
    text = ' '.join(index2word.get(i) for i in int_seq)
    return text

#Add PAD to train_data and test_data
train_data = keras.preprocessing.sequence.pad_sequences(train_data, maxlen = max_length, padding = 'post', value = word2index['<PAD>'])
test_data = keras.preprocessing.sequence.pad_sequences(test_data, maxlen = max_length, padding = 'post', value = word2index['<PAD>'])
