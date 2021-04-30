import pandas as pd
import numpy as np
from os.path import join,exists
from os import makedirs
from glob import glob
from sys import path
path.insert(0,'../src/')

from chemicalencoder import ChemicalEncoder

from tensorflow.keras.layers import Input, Embedding, Flatten, Dense,Concatenate,Masking
from tensorflow.keras.layers import Dropout,Reshape,Average,Minimum, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model as tf_load_model
from tensorflow.keras.backend import sin as tf_sin

class Chemulator:
    def __init__(self,autoencoder_dir=None,species=None):

        self.physics_labels=['gas_temp','gas_density','radfield','zeta','coldens','h2col','ccol',"metallicity"]

        if autoencoder_dir is None:
            self.encode_chem=False
            self.chem_labels=species
            self.species=species
        else:
            #neural nets to encode and decode abundances     
            self.autoencoder=ChemicalEncoder(["H","H+","H2","E-"],drop_hx=False)
            self.autoencoder.load_autoencoder(autoencoder_dir)
            #get size of encoded chemical layer and the list of non-zero nodes
            self.chem_labels=[f"Chem_{i+1}" for i in range(self.autoencoder.encoded_size)]
            self.encode_chem=True
            self.species=self.autoencoder.species

        cs=np.asarray([1 if  "C" in x else 0 for x in self.species])
        cs=cs+np.asarray([1 if  "C2" in x else 0 for x in self.species])
        self.c_idxs=cs+np.asarray([2 if  "C3" in x else 0 for x in self.species])
        self.models=[]
        self.model_names=[]

    def create_single_model(self,n_inputs,n_outputs,layers,activation="relu",dropout=0.8,regularizer=0.00,batch_size=128,
        loss_func="mse",noise=0.0):
        name="-".join(f"{layer}" for layer in layers)+activation
        while name in self.model_names:
            name=name+"I"
        if activation=="sin":
                    activation=tf_sin


        model = Sequential(name=name)
        model.add(Input(shape=(n_inputs,)))
        if noise>0.0:
            model.add(GaussianNoise(noise))
        for layer in layers:
            if layer=="drop":
                model.add(Dropout(dropout))
            else:
                model.add(Dense(layer, activation=activation))

        model.add(Dense(n_outputs,activation='sigmoid'))        

        model.compile(loss=loss_func, optimizer='adam')
        self.models.append(model)
        self.model_names.append(name)

    def create_ensemble(self):
        if len(self.models)>1:
            model_input = Input(shape=(self.models[0].input_shape[1],))
            outputs = [model(model_input) for model in self.models]
            y = Average()(outputs)
            self.model = Model(model_input, y, name='ensemble')
        else:
            self.model=self.models[0]

    def load_model(self,model_folder):
        models=glob(join(model_folder,"*.h5"))
        self.models=[tf_load_model(model,custom_objects={"sin":tf_sin},compile=False) for model in models]
        self.model_names=[]
        for i,model in enumerate(self.models):
            while self.models[i].name in self.model_names:
                self.models[i]._name=self.models[i]._name+"I"
            self.model_names.append(self.models[i]._name)
        self.model_names=[model.name for model in self.models]
        self.create_ensemble()
        self.input_scaling_summary=pd.read_csv(join(model_folder,"input_scaling.csv"),index_col=0) 
        self.output_scaling_summary=pd.read_csv(join(model_folder,"output_scaling.csv"),index_col=0) 

    def save_model(self,model_folder):
        if not exists(model_folder):
            makedirs(model_folder)
        self.input_scaling_summary.to_csv(join(model_folder,"input_scaling.csv"))
        self.output_scaling_summary.to_csv(join(model_folder,"output_scaling.csv"))
        for model in self.models:
            model.save(join(model_folder,f"{model.name}.h5"))
        


    def predict(self,input_data):
        output=self.model.predict(input_data)
        output=pd.DataFrame(columns=["gas_temp","dust_temp"]+self.chem_labels,data=output)
        return output

    def predict_multiple_timesteps(self,input_data,n_steps):
        input_data=input_data.copy()
        for i in range(n_steps):
            output=self.model.predict(input_data.values)
            input_data.loc[:,"gas_temp"]=output[:,0]
            input_data.loc[:,self.chem_labels]=output[:,2:]
        output=pd.DataFrame(columns=["gas_temp","dust_temp"]+self.chem_labels,data=output)
        return output

    def prepare_inputs(self,input_data,learn_scaling=False):
        '''
            Take an (nsamples,nphysics) array of physical conditions
            and an (nsamples,nspecies) array of abundances. Log parameters
            that vary over orders of magnitude, min-max scale them and encoded abundances.

            returns (nsamples,n_inputs) array of scaled variables for emulator.
    '''
        input_data["metallicity"]=(input_data[self.species].values*self.c_idxs).sum(axis=1)
        input_data["metallicity"]=input_data["metallicity"]/2.6e-4
        try:
            inputs=np.log10(input_data[self.physics_labels].reset_index(drop=True))
        except:
            print("missing physical parameters or incorrect labels for those parameters in input data")
            print("the following columns are required")
            print(self.physics_labels)

        #create dataframe from log physics variables and encoded chemistry
        inputs=inputs.merge(
            pd.DataFrame(columns=self.chem_labels,data=self.prepare_chemistry(input_data)),
            left_index=True,right_index=True)

        #either get the min-max scaling from this data or load it
        if learn_scaling:
            summary=inputs.describe().transpose()
            self.input_scaling_summary=summary
        else:
            summary=self.input_scaling_summary

        #min-max scale
        inputs=inputs-summary["min"]
        inputs=inputs/(summary["max"]-summary["min"])
        #clip inputs to min/max derived from training.
        inputs[inputs<0]=0
        inputs[inputs>1]=1.0
        return inputs

    def prepare_outputs(self,output_data,learn_scaling=False,encode_abundances=True):
        outputs=output_data[["gas_temp","dust_temp"]].reset_index(drop=True)
        outputs=np.log10(outputs)

        outputs=pd.concat([outputs,pd.DataFrame(columns=self.chem_labels,data=self.prepare_chemistry(output_data))],axis=1)
 

        #we want to scale outputs 0-1 too but I want same scaling for input and output chemistry so pull from input scaling tabloe
        if learn_scaling:
            summary=outputs.describe().transpose()
            summary.loc[self.chem_labels,:]=self.input_scaling_summary.loc[self.chem_labels,:]
            self.output_scaling_summary=summary
        else:
            summary=self.output_scaling_summary

        #min-max scale
        outputs=outputs-summary["min"].values
        outputs=outputs/(summary["max"]-summary["min"]).values
        #clip outputs. If a value is entered lowered than training min, set to min
        outputs[outputs<0]=0.0
        outputs[outputs>1]=1.0
        return outputs

    def recover_real_values(self,outputs):
        summary=self.output_scaling_summary
        outputs=outputs.values*(summary["max"]-summary["min"]).values
        outputs=outputs+summary["min"].values

        if self.encode_chem:
            df=self.recover_chemistry(outputs[:,2:])
        df=pd.DataFrame(columns=self.species,data=df)
        df["gas_temp"]=10.0**outputs[:,0]
        df["dust_temp"]=10.0**outputs[:,1]
        return df[["gas_temp","dust_temp"]+list(self.species)]

    def prepare_chemistry(self,chem_data):
        if self.encode_chem:
            chem_data=self.autoencoder.prepare_inputs(chem_data)
            chem_data=self.autoencoder.encode(chem_data)
        else:
            chem_data=np.log10(chem_data)
            chem_data=np.where(chem_data<-20,-20,chem_data)
        return chem_data

    def recover_chemistry(self,chem_data):
        if self.encode_chem:
            chem_data=self.autoencoder.decode(chem_data)
            chem_data=self.autoencoder.recover_abundances(chem_data)
        else:
            chem_data=10.00*chem_data
        return chem_data