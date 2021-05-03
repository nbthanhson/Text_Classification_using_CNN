from keras.layers import Embedding, Conv1D, MaxPool1D, GlobalAvgPool1D, GlobalMaxPool1D, Flatten, Dense

vocal_size = len(word2index)
embedding_dim = 50

model = keras.Sequential()

#Add an Embedding layer
model.add(Embedding(input_dim = vocal_size, output_dim = embedding_dim, input_length = max_length))

#Add a Convolutional layer
model.add(Conv1D(filters = 128, kernel_size = 3, activation = 'relu', input_shape = (None, max_length, embedding_dim)))

#Add a Max Pooling layer
model.add(MaxPool1D(pool_size = 2))

#Add a GlobalAvgPool1D layer
model.add(GlobalAvgPool1D())

#Add the output layer
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#Display model summary
model.summary()
