import pandas as pd
from tensorflow import keras
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class AbundanceGenerator(keras.utils.Sequence):
    def __init__(self, abundance_file,species,batch_size=32, shuffle=True,to_fit=True):
        'Initialization'
        self.abundance_file=abundance_file
        self.dim = len(species)
        self.store = pd.HDFStore(abundance_file)#
        self.species=species
        self.n_examples=self.store.get_storer('df').shape[0]
        self.batch_size = batch_size
        self.n_batches=int(np.floor(self.n_examples / self.batch_size))
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(self.n_batches)
      if self.shuffle == True:
          np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        X=self.store.select(key="df",start=index*self.batch_size,stop=(index+1)*self.batch_size)
        return X[self.species].values,X[self.species].values

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_batches