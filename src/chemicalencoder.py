from os.path import join,exists
from os import makedirs
from os import environ
import numpy as np
import pandas as pd


#tensorflow stuff
environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense,Concatenate,Masking,Dropout,Reshape,GaussianNoise
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers,Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.activations import relu

class ChemicalEncoder:
    def __init__(self,species=None,min_val=-20,drop_hx=True):
        self.drop_hx=drop_hx
        self.min_val=min_val
        if species is not None:
            self.species=np.asarray(species)
            self.set_species_variables()



    def create_model(self,layers,activation="relu",dropout=True,optimizer="adam",
                 batch_size=32,loss_func="mse",repeat=False,noise=0.0):
    
        if activation=="leaky":
            activation=tf.keras.layers.LeakyReLU(alpha=0.1)
        if activation=="sin":
            activation=tf.keras.backend.sin

        encoder_layers=layers[:np.argmin(layers)+1]
        decoder_layers=layers[np.argmin(layers):]
        self.encoded_size=np.min(layers)
        
        encoder_input = Input(shape=(len(self.encoded_species),))
        encoder_output=Dense(encoder_layers[0],activation=activation)(encoder_input)
        for layer in encoder_layers[1:]:
            encoder_output=Dense(layer,activation=activation)(encoder_output)
        self.encoder=Model(encoder_input,encoder_output,name="encoder")
        
        decoder_input = Input(shape=(self.encoded_size,))
        if noise>0.0:
            decoder_output=GaussianNoise(noise)(decoder_input)
            n=1
        else:
            decoder_output=Dense(decoder_layers[1],activation=activation)(decoder_input)
            n=2
        for layer in decoder_layers[n:]:
            decoder_output=Dense(layer,activation=activation)(decoder_output)
        decoder_output=Dense(len(self.encoded_species),activation=activation)(decoder_output)
        self.decoder=Model(decoder_input,decoder_output,name="decoder")
        
        
        autoencoder_input = Input(shape=(len(self.encoded_species),))
        encoded_chem = self.encoder(autoencoder_input)
        decoded_chem = self.decoder(encoded_chem)
        if repeat:
            encoded_chem = self.encoder(decoded_chem)
            decoded_chem = self.decoder(encoded_chem)  
        
        self.autoencoder = Model(autoencoder_input, decoded_chem, name="autoencoder")

        if optimizer=="sgd":
            optmizer=SGD(lr=0.005,momentum=0.9)
        self.autoencoder.compile(loss=loss_func, optimizer=optimizer)

    def save_autoencoder(self,model_folder):
        if not exists(model_folder):
            makedirs(model_folder)
        self.encoder.save(join(model_folder,"encoder.h5"))
        self.decoder.save(join(model_folder,"decoder.h5"))
        pd.Series(self.species).to_csv(join(model_folder,"species.csv"),index=False,header=False)

    def load_autoencoder(self,model_folder):
        self.species=np.loadtxt(join(model_folder,"species.csv"),dtype=str,comments=None)
        self.set_species_variables()
        self.encoder=load_model(join(model_folder,"encoder.h5"),compile=False)
        self.decoder=load_model(join(model_folder,"decoder.h5"),compile=False)
        self.encoded_size=self.decoder.layers[0].output_shape[0][1]
        autoencoder_input = Input(shape=(len(self.encoded_species),))
        encoded_chem = self.encoder(autoencoder_input)
        decoded_chem = self.decoder(encoded_chem)

        self.autoencoder = Model(autoencoder_input, decoded_chem, name="autoencoder")

    def set_species_variables(self):
        drop_species=["E-"]
        if self.drop_hx:
            drop_species.append("H+")
        self.encode_idx=[i for i in range(len(self.species)) if self.species[i] not in drop_species]
        self.encoded_species=self.species[self.encode_idx]
        #finally the indices of species for sums
        self.oxygen=[i for i in range(len(self.species)) if "O" in self.species[i]]
        self.ions=[i for i in range(len(self.species)) if "+" in self.species[i]]

        self.h_indx=np.where(self.species=="H")[0][0]
        self.h2_indx=np.where(self.species=="H2")[0][0]
        self.hx_indx=np.where(self.species=="H+")[0][0]
        self.e_indx=np.where(self.species=="E-")[0][0]

        hs=np.asarray([1 if "H" in x.replace("HE","") else 0 for x in self.species])
        hs=hs+np.asarray([1 if "H2" in x else 0 for x in self.species])
        hs=hs+np.asarray([2 if "H3" in x else 0 for x in self.species])
        self.hs=hs+np.asarray([3 if "H4" in x else 0 for x in self.species])

    def prepare_inputs(self,df):
        try:
            df=df[self.encoded_species]
        except:
            print("species list used to initialize autoencoder does not match input data")
            print("input data must be a dataframe with at least one column for each species")
            print("autoencoder species list:")
            print(self.species)
            return -1

        #logscale abundances so we don't favour large abundances only
        #set a minimum value so that we don't try to be accurate on values that are basically 0
        df=np.where(df<10.0**self.min_val,10.0**self.min_val,df)
        df=np.log10(df)
        df=pd.DataFrame(columns=self.encoded_species,data=df)
        
        #we'll then scale values so they go from 0 to 1
        df=(df+np.abs(self.min_val))/np.abs(self.min_val)

        return df

    def encode(self,prepared_data):
        encoded_data=self.encoder.predict(prepared_data)
        return encoded_data

    def decode(self,encoded_data):
        prepared_data=self.decoder.predict(encoded_data)
        return prepared_data

    def recover_abundances(self,abundances):
        '''
            Turn an array of scaled abundances into an array of full abundances.
            args:
                prepared_data: log and scaled abundances of species in self.encoded_species
            returns:
                array of abundances for all species in self.species.

        '''

        #unscale abundances
        abundances=(np.abs(self.min_val)*(abundances))-np.abs(self.min_val)
        abundances=np.where(abundances>0.0,0.0,abundances)
        abundances=10.00**abundances

        #repad encoded abundances
        for i in range(len(self.species)):
            if i not in self.encode_idx:
                abundances=np.insert(abundances,i,0.0,axis=1)
               
        #Stop any overly large H or H2 values
        abundances[:,self.h_indx]=np.where(abundances[:,self.h_indx]>1.0,1.0,abundances[:,self.h_indx])
        abundances[:,self.h2_indx]=np.where(abundances[:,self.h2_indx]>0.5,0.5,abundances[:,self.h2_indx])
        
        #calculate excess H by summing all H and subtracting 1
        H_excess=(abundances*self.hs).sum(axis=1)-1.0
        H_excess=np.where(H_excess<0.0,0.0,H_excess)

        # #remove H excess from H or H2 depending which has more of the H nuclei
        # more_h=abundances[:,self.h_indx]>2.0*abundances[:,self.h2_indx]
        # abundances[:,self.h_indx]=np.where(more_h,abundances[:,self.h_indx]-H_excess,abundances[:,self.h_indx])
        # abundances[:,self.h2_indx]=np.where(~more_h,abundances[:,self.h2_indx]-H_excess,abundances[:,self.h2_indx])

        #H+ is 1-(sum of all H bearing species)
        #If it's negative, there's too much H so reduce H2 and set H+ to zero
        if self.drop_hx:
            abundances[:,self.hx_indx]=1.0-(abundances*self.hs).sum(axis=1)

        #electron abundance is sum of ions
        electrons=abundances[:,self.ions].sum(axis=1)
        abundances[:,self.e_indx]=electrons
        min_val=10.0**self.min_val
        abundances=np.where(abundances<min_val,min_val,abundances)
        abundances=pd.DataFrame(columns=self.species,data=abundances)
        return abundances
