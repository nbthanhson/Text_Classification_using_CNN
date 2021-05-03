from tensorflow import keras
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

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#Compile the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Early stop training
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1)

#Save the best model
best_model = ModelCheckpoint(filepath = 'Text_Classification_bestmodel.h5', monitor = 'val_loss', verbose = 1, save_best_only = True)

#Reduce learning rate
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 5, verbose = 1, min_lr = 0.001)

#Start training
model_history = model.fit(train_data, train_labels,
                          batch_size = 512, epochs = 50, validation_split = 0.3,
                          callbacks = [early_stopping, best_model, reduce_lr],
                          shuffle = True)

#Load the best model
keras.models.load_model(filepath = 'Text_Classification_bestmodel.h5')

model.evaluate(test_data, test_labels)
