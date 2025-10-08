#!/usr/bin/env python
# coding: utf-8
import os
import h5py  # Large size Hierachical Data Format (HDF) file
import cv2
import re
import numpy as np
import tensorflow as tf
import pandas as pd
from joblib import Parallel, delayed #Parallel computing
from tqdm import tqdm
from FiltNet_CNNs import FiltNet40,modelmake #import the Neural network architecture
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16,VGG19,ResNet101,InceptionV3,EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback,ModelCheckpoint, EarlyStopping, TensorBoard
import tensorflow.keras.backend as kb 
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

#================================================================================================#
# Initialization
#================================================================================================#
Features_Num = int(308) #Number of features
ImgSize = 256
Base_dir = 'FiltNet-for-rapidly-predicting-fibrous-filter-media-properties' #Base directory
suffix = '_10066d' #File name
MinMax_dir = f'{Base_dir}/FiltNet_training/MinMax/'
Log_dir = f'{Base_dir}/Logs/' #Log directory
List_path = f'{Base_dir}/HDF/List/' #List path
TrainedModel_path = f'{Base_dir}/HDF/Trained/'
VarName_path = f'{Base_dir}/VarNames.txt' #Variable name path
Img_evaluation_dir = f'{Base_dir}/FiltNet_training/General_Evaluation/'
'''
14 single-value features, 
50 solidity distribution along x-axis, 
25 solidity distribution along z-axis, 
200 pore size distritbution, 
19 filtration efficiencies
'''    
data_indices = [14, 50, 25, 200, 19]
#================================================================================================#
# Data preparation
#================================================================================================#
# RGB Image data generation
def BGR_img_gen(subfolder_path,ImgSize=ImgSize,NZ_value=None):
    subfolder = os.path.basename(os.path.normpath(subfolder_path))
    BGR_img = np.zeros((ImgSize, ImgSize, 3), dtype=np.uint8)

    # Compute NZ_value
    if NZ_value is None:
        t_value = float(re.search(r"t_(\d+\.\d+)", subfolder).group(1))
        VL_value = float(re.search(r"VL_(\d+\.\d+)", subfolder).group(1))
        NZ_value = int(np.ceil(t_value / VL_value))
    else:
        NZ_value = int(NZ_value)
    
    new_size = (ImgSize,ImgSize)
    for file in os.listdir(subfolder_path):
        if file.endswith(".png") and "matched" in file.lower():
            image_path=os.path.join(subfolder_path, file)
            basename=os.path.splitext(os.path.basename(image_path))[0]
            
            ##image process to remove the black border according to the filter thickness and voxel sizes        
            img=cv2.imread(image_path)
            if "x_y" in basename:
                if 50 <= NZ_value <= 128:
                    img=img[4:546, 4:546]
                elif 129 <= NZ_value <= 297:
                    img=img[3:547, 3:547]
                else:
                    img=img[2:548, 2:548]
                NLM=cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)  # Non Local Mean filter                    
                gray=cv2.cvtColor(NLM, cv2.COLOR_BGR2GRAY) # convert into gray image
                reSizeImg = cv2.resize(gray,new_size,interpolation=cv2.INTER_AREA)
                BGR_img[:, :, 2]=reSizeImg # save image into a R channel
            elif "x_z" in basename:
                img=img[1:549, 1:549]
                NLM=cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)                 
                gray=cv2.cvtColor(NLM, cv2.COLOR_BGR2GRAY)
                reSizeImg = cv2.resize(gray,new_size,interpolation=cv2.INTER_AREA)
                BGR_img[:, :, 1]=reSizeImg # save image into a G channel
            elif "y_z" in basename:
                img=img[1:549, 1:549]
                NLM=cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)                
                gray=cv2.cvtColor(NLM, cv2.COLOR_BGR2GRAY)
                reSizeImg = cv2.resize(gray,new_size,interpolation=cv2.INTER_AREA)
                BGR_img[:, :, 0]=reSizeImg # save image into a B channel
            else:
                print("Error")
    return BGR_img

def BGR_img_prep(folder_path, ImgSize=ImgSize, NZ_value=None):
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        subfolder_path_components = subfolder_path.split(os.path.sep)

        bgr = BGR_img_gen(subfolder_path,ImgSize=ImgSize,NZ_value=NZ_value)

        #save feature map
        img_folder_path_1 = subfolder_path_components[:-2]
        img_folder_path = os.path.sep.join(img_folder_path_1) + f'/Matched_Feature_maps_smooth_{ImgSize}'
        imgName = subfolder_path_components[-1] + '.png'
        imgPath=os.path.join(img_folder_path, imgName)
        os.makedirs(img_folder_path, exist_ok=True)
        cv2.imwrite(imgPath, bgr)
        print(f"Save images for {subfolder}.")


#================================================================================================#
# Create HDF files (associate img with its simulation data)
def write_rgb_images_to_h5(cvs_file, hdf_file, img_size, img_folder):
    # read the file name from the last column of the csv file
    csv_file_path = cvs_file
    hdf_file_path = hdf_file
    img_folder_path = img_folder
    df = pd.read_csv(csv_file_path)

    with h5py.File(hdf_file_path, "w") as hdf5_file:
        x_dataset = hdf5_file.create_dataset("X", shape=(len(df), img_size, img_size, 3), dtype = 'float32')
        y_dataset = hdf5_file.create_dataset("Y", shape=(len(df),Features_Num,1),  dtype = 'float32')
        x2_dataset = hdf5_file.create_dataset("X2", shape=(len(df), 1), dtype='float32')  # Fifth column as metadata
        
        for index, row in df.iterrows():
            image_path = os.path.join(img_folder_path, row['Img_name'])
            img = cv2.imread(image_path) 
            img_array = np.array(img)
            img_array = img_array / 255.0 # Normalize the grayscale of each piexl

            x_dataset[index, :, :, :] = img_array

            # Extracting the third column as metadata
            x2_dataset[index, 0] = row.iloc[2]
            y_values = pd.concat([row.iloc[1:2],row.iloc[4:6],row.iloc[10:11],row.iloc[12:13],row.iloc[14:22],row.iloc[23:318]]).to_numpy().astype('float32').reshape(Features_Num, 1)
            y_dataset[index, ...] = y_values

#================================================================================================#
# Data augmentation
def img_augmentation(imgs, X2, Y):
    """
     1. Horizontal flip of images x_y and x_z
     2. Vertical flip of images x_y and y_z
     3. Vertical flip of images x_z and y_z  
    """
    X2 = np.concatenate((X2,X2,X2,X2), axis=0)
    Y1 = np.copy(Y)
    Y1[:,14:64,:] = np.flip(Y1[:,14:64,:], axis=1)
    Y2 = np.copy(Y)
    Y2[:,64:89,:] = np.flip(Y2[:,64:89,:], axis=1)
    Y = np.concatenate((Y,Y1,Y,Y2), axis=0)
    
    augmented_imgs_x_y = np.copy(imgs)
    augmented_imgs_y_z = np.copy(imgs)
    augmented_imgs_x_z = np.copy(imgs)
    
    # Horizontal flip of images x_y and x_z
    for i in range(len(augmented_imgs_x_y)):
        augmented_imgs_x_y[i, ..., 2] = np.flip(augmented_imgs_x_y[i, ..., 2], axis=1)
        augmented_imgs_x_y[i, ..., 1] = np.flip(augmented_imgs_x_y[i, ..., 1], axis=1)

    # Vertical flip of images x_y and y_z
    for j in range(len(augmented_imgs_y_z)):
        augmented_imgs_y_z[j, ..., 2] = np.flip(augmented_imgs_y_z[j, ..., 2], axis=0)
        augmented_imgs_y_z[j, ..., 0] = np.flip(augmented_imgs_y_z[j, ..., 0], axis=1)
        
    for k in range(len(augmented_imgs_x_z)):
        augmented_imgs_x_z[k, ..., 1] = np.flip(augmented_imgs_x_z[k, ..., 1], axis=0)
        augmented_imgs_x_z[k, ..., 0] = np.flip(augmented_imgs_x_z[k, ..., 0], axis=0)
    
    imgs = np.concatenate((imgs,augmented_imgs_x_y,augmented_imgs_y_z,augmented_imgs_x_z), axis=0)
    return imgs,X2,Y

#================================================================================================#
# Training inputs preparation
#================================================================================================#

# Get the x-index of the input data
# Get the mix and max values in each row of the input data
def prep(Data,ModelType=40):
    print('Checking the data for outliers. Please wait....')
    List = []
    
    with h5py.File(Data, 'r') as f: 
        imgs = f['X'][:,...]
        X2 = f['X2'][:,...]
        Y = f['Y'][:,...]
        
        X2 = (X2 * 400.0 / ImgSize).astype('float32') # Voxel size * NX(or NY/NZ) / Img size
        X2 = np.log10(X2)
        X2_MIN, X2_MAX = np.min(X2), np.max(X2)
        
        Y[:,-19:] = np.log10(Y[:,-19:])
        imgs,X2,Y = img_augmentation(imgs, X2, Y)

        length = imgs.shape[0]
        MIN = np.ones((Y.shape[1], 1))*1e11 
        MAX = -MIN 
        counter = 0
        
        for I in range(length):
            y = Y[counter, ...].astype('float32')
            y_s = y[0:14]
            if np.isnan(y).sum() > 0 or np.isinf(y).sum() > 0 or (y == 0).sum() > 0: # Skip rows with NaN, infinity, or zero in certain features
                pass
            else:
                y[0:14] = np.log10(y[0:14]) # Apply log scaling to avoid large value discrepancies
                # Update max/min values for each feature
                MAX = np.maximum(MAX, y)
                MIN = np.minimum(MIN, y)
                List = np.append(List, counter)

            if counter % 100 == 0:
                print('checking sample' + str(counter))
            counter = counter + 1

        
        for I in range(1,5):
            # get the maximum value of each mulit-value parameters
            MAX[np.sum(data_indices[0:I]): np.sum(data_indices[0: I+ 1])] = np.max(MAX[np.sum(data_indices[0:I]): np.sum(data_indices[0: I+ 1])]) 
            # get the minimum value of each mulit-value parameters                                                             
            MIN[np.sum(data_indices[0:I]): np.sum(data_indices[0: I+ 1])] = np.min(MIN[np.sum(data_indices[0:I]): np.sum(data_indices[0: I+ 1])])
            
    np.save(f'{MinMax_dir}minmax_model{ModelType}{suffix}.npy', [MIN, MAX])
    np.save(f'{MinMax_dir}minmax_res_model{ModelType}{suffix}.npy', [X2_MIN, X2_MAX])
    
    with h5py.File(f'{List_path}list_{ModelType}{suffix}.h5', 'w') as file:
        file.create_dataset('List', data=List)
    return List

#================================================================================================#
# Reading the list from the HDF5 File
def load_list(file_path):
    with h5py.File(file_path, 'r') as file:
        List = list(file['List'][:])
    return List

#================================================================================================#
# Generate data (retrain or reload trained model) to work with a multi-head output structure
# Separate y_batch target array into multiple outputs: orientation tensors y_batch[:, 10:13], other output
def gener(batch_size,Data,List,MIN,MAX,X2_MIN,X2_MAX):
    with h5py.File(Data,'r') as f:
        length=len(List)
        samples_per_epoch = length
        number_of_batches = samples_per_epoch // batch_size
        
        imgs = f['X'][:,...]
        X2 = f['X2'][:,...]
        Y = f['Y'][:,...]
        
        # Normalize metadata according to voxel and pixel size
        X2 = (X2 * 400.0 / ImgSize).astype('float32')
        X2 = np.log10(X2)  # Logrithm of metadata

        #Data augmetation   
        imgs,X2,Y = img_augmentation(imgs, X2, Y)

        counter=0
        while True:
            # Get current batch
            batch_indices = np.int32(np.sort(List[batch_size * counter : batch_size * (counter + 1)]))
            X_batch = imgs[batch_indices, ...].astype('float32')
            X2_batch = X2[batch_indices, ...]
            y_batch = Y[batch_indices, ...].astype('float32')

            # normalize the metadata
            X2_batch = (X2_batch - X2_MIN) / (X2_MAX - X2_MIN)
            
            # Normalize targets (Y)
            y_batch=np.reshape(y_batch,(y_batch.shape[0],y_batch.shape[1])) # (100, 215)
            y_batch[:, 0:14] = np.log10(y_batch[:, 0:14]) # Logrithm of single-value features
            y_batch[:, -19:] = np.log10(y_batch[:, -19:])
            Min=np.tile(np.transpose(MIN),(batch_size,1)) # 转置 (100, 215)
            Max=np.tile(np.transpose(MAX),(batch_size,1)) # (100,215)
            valid = (Max - Min != 0) 
            y_batch[valid] = (y_batch[valid] - Min[valid]) / (Max[valid] - Min[valid])
            
            # Split targets into orientation, FFE, and other outputs
            orientation_output = y_batch[:, [10, 11, 12]]  # Select columns 11, 12, and 13 (0-based index)
            FFE_output = y_batch[:, -19:]   # Select columns 11, 12, and 13 (0-based index)
            cols_to_remove = [10, 11, 12] + list(range(y_batch.shape[1] - 19, y_batch.shape[1]))
            other_output = np.delete(y_batch, cols_to_remove, axis=1)
             
            # Shuffle batch data
            ids = np.arange(len(y_batch))
            np.random.shuffle(ids)
            X_batch=X_batch[ids,...]
            y_batch=y_batch[ids,...]
            X2_batch=X2_batch[ids,...]
            orientation_output = orientation_output[ids, ...]
            FFE_output = FFE_output[ids, ...]
            other_output = other_output[ids, ...]
            
            yield [X_batch, X2_batch], {"orientation_output": orientation_output, 
                                        "FFE_output": FFE_output, 
                                        "other_output": other_output}
            
            counter += 1
            if counter >= number_of_batches: #restart counter to yeild data in the next epoch as well
                counter = 0
                np.random.shuffle(List)

#================================================================================================#
## Split data
## Train data : Evaluate data: Test data = 0.8 : 0.1 : 0.1
def splitdata(List):
    N=np.int32([0,len(List)*.8,len(List)*.9,len(List)])
    TrainList=List[N[0]:N[1]]
    EvalList=List[N[1]:N[2]]
    TestList=List[N[2]:N[3]]
    return TrainList, EvalList, TestList

#================================================================================================#
# Get the shape of the hdf file
def hdf_shapes(Name, Argument):
    # Fields is list of hdf file fields
    Shape = [[] for _ in range(len(Argument))]
    with h5py.File(Name, 'r') as f:
        for I in range(len(Argument)):
            Shape[I] = f[Argument[I]].shape
    return Shape

#================================================================================================#
# Get the string of the current times s
def nowstr():
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%d-%b-%Y %H.%M.%S")
 
#================================================================================================#
# Train model
#================================================================================================#

def log_to_file(message, log_path):
    with open(log_path, "a") as f:
        f.write(message + "\n")

class MyCallback(Callback):
    def __init__(self, log_path, log_interval=1):
        self.log_path = log_path
        self.log_interval = log_interval
        self.val_loss_orientation = None
        self.val_loss_FFE = None
        self.val_loss_other = None

    def on_batch_end(self, batch, logs=None):
        if batch % self.log_interval == 0:
            train_loss_orientation = logs.get('orientation_output_loss', 0)
            train_loss_FFE = logs['FFE_output_loss']
            train_loss_other = logs.get('other_output_loss', 0)
            val_loss_orientation = self.val_loss_orientation or train_loss_orientation
            val_loss_FFE = self.val_loss_FFE or train_loss_FFE
            val_loss_other = self.val_loss_other or train_loss_other

            log_to_file(
                f"{train_loss_orientation:<30}{train_loss_FFE:<30}{train_loss_other:<30}"
                f"{val_loss_orientation:<30}{val_loss_FFE:<30}{val_loss_other}",
                self.log_path
            )

    def on_test_batch_end(self, batch, logs=None):
        self.val_loss_orientation = logs.get('orientation_output_loss')
        self.val_loss_FFE = logs.get('FFE_output_loss')
        self.val_loss_other = logs.get('other_output_loss')

    def on_epoch_end(self, epoch, logs=None):
        log_to_file(f"# End of epoch {epoch + 1}", self.log_path)

def trainmodel(DataName,
               TrainList,
               EvalList,
               retrain=False,
               reload=False,
               fine_tuning=False,
               epochs=100,
               batch_size=100,
               epochs_2 = 10,
               ModelType=40,
               log_interval=1
               ):
    
    model_name = f'Model{ModelType}FineTune{suffix}.h5'
    SaveName = TrainedModel_path + model_name

    # load Min and Max from the .npy file
    MIN, MAX = np.load(f'{MinMax_dir}minmax_model{ModelType}{suffix}.npy')
    X2_MIN, X2_MAX = np.load(f'{MinMax_dir}minmax_res_model{ModelType}{suffix}.npy')
    

    # Log file
    timestr = nowstr()
    LogName = f'log_{timestr}_Model{ModelType}{suffix}.txt'
    LogPath = os.path.join(Log_dir, LogName)

    # Write log header
    with open(LogPath, "w") as f:
        f.write(f'# Path to train file:\n{DataName}\n')
        f.write(f'# Start time:\n{timestr}\n')
        header = ('# Training loss (Orientation)'.ljust(30) +
                  'Training loss (FFE)'.ljust(30) +
                  'Training loss (Other)'.ljust(30) +
                  'Val loss (Orientation)'.ljust(30) +
                  'Val loss (FFE)'.ljust(30) +
                  'Val loss (Other)')
        f.write(header + '\n')

    # Model setup
    INPUT_SHAPE, INPUT_SHAPE_2,OUTPUT_SHAPE = hdf_shapes(DataName, ('X', 'X2', 'Y'))
    OUTPUT_SHAPE = [1, 1]

    # Callbacks
    callbacks_list = [
        MyCallback(LogPath, log_interval=log_interval),
        ModelCheckpoint(SaveName, monitor='loss', save_best_only=True, verbose=1, mode='min', save_freq=50),
        #EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    ]
    
    # end of callbacks
    model = modelmake(INPUT_SHAPE, INPUT_SHAPE_2, OUTPUT_SHAPE, ModelType)

    if retrain: 
        if reload:
            model.load_weights(
                SaveName
            )  # load the model weights from the specified 'SaveName'

        # Initial training
        model.fit(gener(batch_size, DataName, TrainList, MIN, MAX), # 
                  epochs=epochs,
                  steps_per_epoch=len(TrainList) // batch_size,
                  validation_data=gener(batch_size, DataName, EvalList, MIN,
                                        MAX),
                  validation_steps=len(EvalList) // batch_size,
                  callbacks=callbacks_list)
        
        if fine_tuning:
            for layer in model.layers:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
            if ModelType == 38:
                model.compile(
                    optimizer=Adam(learning_rate=1e-5),
                    loss={'orientation_output': 'mse','FFE_output': 'mse', 'other_output': 'mse'},
                    loss_weights={'orientation_output': 1, 'FFE_output': 0.1, 'other_output': 0.1},
                    metrics={'orientation_output': 'mse', 'FFE_output': 'mse','other_output': 'mse'}                     
                )
            elif ModelType == 40:
                model.compile(
                    optimizer=Adam(learning_rate=1e-5),
                    loss={'orientation_output': 'mse','FFE_output': 'mse', 'other_output': 'mse'},
                    loss_weights={'orientation_output': 0.5, 'FFE_output': 1.0, 'other_output': 1.0},
                    metrics={'orientation_output': 'mse', 'FFE_output': 'mse','other_output': 'mse'}                     
                )
            elif ModelType == 42:
                model.compile(
                    optimizer=Adam(learning_rate=1e-5),
                    loss={'orientation_output': 'mse','FFE_output': 'mse', 'other_output': 'mse'},
                    loss_weights={'orientation_output': 1.0, 'FFE_output': 1.0, 'other_output': 1.0},
                    metrics={'orientation_output': 'mse', 'FFE_output': 'mse','other_output': 'mse'}                     
                )
            else:
                model.compile(optimizer=Adam(learning_rate=1e-5),  loss='mse', metrics=['mse'])
                            
            model.fit(gener(batch_size, DataName, TrainList, MIN, MAX), # 
                      epochs=epochs_2,
                      steps_per_epoch=len(TrainList) // batch_size,
                      validation_data=gener(batch_size, DataName, EvalList, MIN,
                                            MAX),
                      validation_steps=len(EvalList) // batch_size,
                      callbacks=callbacks_list)
        model.save_weights(SaveName)
    else:
        model.load_weights(SaveName)
    return model

#================================================================================================#
## Parallel computing
def parfor(func,values):
    # example 
    # def calc(I):
    #     return I*2
    # px.parfor(calc,[1,2,3])
    N=len(values) 
    Out = Parallel(n_jobs=-1)(delayed(func)(k) for k in tqdm(range(1,N+1)))
    return Out

#================================================================================================#
# Evaluate the trained model
#================================================================================================#

# Restore the original layout of the reference data from a model output dictionary
def refValue_restore(y_Ori):
    y_orientation_output = y_Ori["orientation_output"]
    y_other_output = y_Ori["other_output"]
    y_FFE_output = y_Ori["FFE_output"]

    num_samples = y_orientation_output.shape[0]
    total_cols = y_other_output.shape[1] + y_orientation_output.shape[1] + y_FFE_output.shape[1]

    y = np.zeros((num_samples, total_cols), dtype=np.float32)

    # Restore other_output before orientation
    y[:, 0:10] = y_other_output[:, 0:10]
    y[:, 10:13] = y_orientation_output
    y[:, 13:-19] = y_other_output[:, 10:]
    y[:, -19:] = y_FFE_output

    return y

#Restore the original layout of the predicted data from a list of outputs
def preValue_restore(y2_Ori):
    y2_orientation_output = y2_Ori[0]
    y2_FFE_output = y2_Ori[1]
    y2_other_output = y2_Ori[2]

    num_samples = y2_orientation_output.shape[0]
    total_cols = y2_other_output.shape[1] + y2_orientation_output.shape[1] + y2_FFE_output.shape[1]

    y2 = np.zeros((num_samples, total_cols), dtype=np.float32)

    # Restore other_output before orientation
    y2[:, 0:10] = y2_other_output[:, 0:10]
    y2[:, 10:13] = y2_orientation_output
    y2[:, 13:-19] = y2_other_output[:, 10:]
    y2[:, -19:] = y2_FFE_output

    return y2

#================================================================================================#
# Load the trained model
def loadmodel(path, model_type=40):
    name = f'Model{model_type}FineTune{suffix}.h5'
    model_path = os.path.join(path, name)
    input_shape = [1, ImgSize, ImgSize, 3]
    input_shape_2 = [1, 1]
    output_shape = [1, Features_Num, 1]
    model = modelmake(input_shape, input_shape_2, output_shape, model_type)
    model.load_weights(model_path)
    return model

#================================================================================================#
# Test model with the R^2 score of single-value features
def testmodel(model,DataName,TestList,ModelType=40):
    MIN,MAX=np.load(f'{MinMax_dir}minmax_model{ModelType}{suffix}.npy')
    X2_MIN, X2_MAX = np.load(f'{MinMax_dir}minmax_res_model{ModelType}{suffix}.npy')

    G=gener(len(TestList),DataName,TestList,MIN,MAX,X2_MIN,X2_MAX)
    L=next(G)
    x, y_Ori = next(G)
    y2_Ori = model.predict(x)

    print('\n# Evaluate on '+ str(len(TestList)) + ' test data')
    model.evaluate(x,y_Ori,batch_size=50)

    #Convert dic to array  
    y = refValue_restore(y_Ori)
    #Convert list to array 
    y2 = preValue_restore(y2_Ori)

    #  Denormalize the predictions
    MIN=np.reshape(MIN,(1,y.shape[1]))
    MAX=np.reshape(MAX,(1,y.shape[1]))
    y=np.multiply(y,(MAX-MIN))+MIN
    y2=np.multiply(y2,(MAX-MIN))+MIN

    y[:, 0:14] = 10**y[:,0:14]
    y[:, -19:] = 10**y[:,-19:]
    y2[:, 0:14] = 10**y2[:,0:14]
    y2[:, -19:] = 10**y2[:,-19:]

    # Show prediction of single-value features
    fig1 = plt.figure(figsize=(45, 25))
    plt.rcParams.update({'font.size': 20})
    with open(VarName_path) as f:
        VarNames = list(f)
    r_squared = []   
    for I in range(14):
        ax = fig1.add_subplot(5, 3, I+1)
        X=y[:,I]
        Y=y2[:,I]
        r2 = r2_score(X, Y)
        r_squared.append(r2)
        plt.scatter(X,Y)
        plt.ylabel('Predicted')
        plt.xlabel('Ground truth')
        plt.tick_params(direction="in")
        plt.text(.5, 0.95, VarNames[I],
             horizontalalignment='center', transform=ax.transAxes)
        plt.text(0.3, 0.9, f'R^2 = {r2:.2f}', ha='center', va='center', transform=plt.gca().transAxes)
        plt.xlim(np.min(X),np.max(X))
        plt.ylim(np.min(Y),np.max(Y))
        plt.plot(X, X, color='green', linestyle='--') # label='y_predict = y_actual

        plt.subplots_adjust(left=0.1,  # Adjust left margin
                        right=0.9,  # Adjust right margin
                        bottom=0.1,  # Adjust bottom margin
                        top=0.9,  # Adjust top margin
                        wspace=0.4,  # Adjust horizontal spacing between subplots
                        hspace=0.4)  # Adjust vertical spacing between subplots

    plt.savefig(f'{Img_evaluation_dir}MultiValue_Features_model{ModelType}{suffix}.png')

#================================================================================================#
# Predict filter properties
#================================================================================================#

def predict(model,A,B,ModelType=40):
    unit_transfer = 1e-6
    MIN,MAX=np.load(f'{MinMax_dir}minmax_model{ModelType}{suffix}.npy')
    A = np.expand_dims(A, axis=0) # Friction RGB image
    B = np.expand_dims(B, axis=0) # Normalized resolution
    X = [A,B]
    y_pre = model.predict(X)
    y = preValue_restore(y_pre)
    
    MIN=np.reshape(MIN,(1,y.shape[1]))
    MAX=np.reshape(MAX,(1,y.shape[1]))
    y=np.multiply(y,(MAX-MIN))+MIN
    y[:,0:14]=10**y[:,0:14]
    y[:,-19:]=10**y[:,-19:]
    y=np.mean(y,axis=0)

    # Unit conversion
    for i in [0, 1, 2, 7, 8, 9]:
        y[i] /= unit_transfer

    single_value_output = y[:14]
    for i in range(len(data_indices) - 1):
        output = np.append(single_value_output, y[np.sum(data_indices[:i + 1]):np.sum(data_indices[:i + 2])])
    return output


#================================================================================================#
#Prettify and Print Results
def prettyresult(vals, pixel_size, file_name, units='um', verbose=1):
    vals = np.squeeze(vals)
    unit_transfer = 1e-6
    res = pixel_size / unit_transfer

    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    with open(VarName_path) as f:
        var_names = [line.strip() for line in f]

    s_values = vals[:14]
    thickness = vals[0]

    with open(file_name, 'w') as f:
        f.write("DeeFilter output results including 14 single-value\nparamters, 4 distributions\n")
        f.write('_' * 50 + '\n')
        f.write('        ### Single-value parameters ###\n')
        f.write('_' * 50 + '\n\n')
        f.write('Properties' + ' ' * 30 + 'Value\n')
        f.write('-' * 50 + '\n')

        for i, val in enumerate(s_values):
            name = var_names[i].replace('(m)', f'({units})') if units == 'um' else var_names[i]
            display_val = np.round(val,3)
            # Compute and write Closed Porosity
            if i == 5:
                closed_porosity = np.round(s_values[i - 2] - s_values[i - 1], 3)
                f.write(f'{"Closed porosity":<40}{closed_porosity}\n')

            # Compute and write Orientation Tensor z
            if i == 12:
                orientation_tensor_z = np.round(1 - 2 * s_values[i - 1], 3)
                display_val = orientation_tensor_z

            # Compute and write Delta_P at 10.5 cm/s
            if i == 13:
                dp = (0.105 * 1.834e-05 * thickness * 1e-6) / val
                f.write(f'{"Delta_P at 10.5 cm/s (Pa)":<40}{dp:.2f}\n')
                display_val = f'{val:.3e}'
            f.write(f'{name:<40}{display_val}\n')

        f.write('\n' + '_' * 50 + '\n')
        f.write('       ### Distributions ###\n')
        f.write('_' * 50 + '\n')

        d = [14, 50, 25, 200, 19]
        for I in range(4):
            multiplier=1
            label = var_names[I + 14].strip()
            xlabel, ylabel = ('In-plane (um)', '       Solidity (-)')
            if I == 1: xlabel = 'Through-plane (um)'
            if I == 2: xlabel, ylabel = 'Pore Diameter (um)', 'Cumulative volume fraction (-)'
            if I == 3: xlabel, ylabel = 'Particle size (um)', 'Filtration efficiency (-)'

            f.write(f'\n\n# {label}\n')
            f.write('-' * 50 + '\n')
            f.write(f'{xlabel:<25}{ylabel}\n')
            f.write('-' * 50 + '\n')

            shift=np.sum(d[0:I+1])
            
            for J in range(d[I+1]):
                if I == 0:
                    multiplier=400 / 50 * res * 256 / 400
                if I == 1:
                    multiplier=thickness / 25
                if I == 2:
                    multiplier=2 * res * 256 / 400
                t=str(np.round((J+1)*multiplier,5))
                if I == 3:
                    if J > 9:
                        multiplier=10
                        t=str(np.round(((J-10)*.01+.02)*multiplier,2))
                    else:
                        t=str(np.round((J*0.01+0.01)*multiplier,2))
                spa=' ' * (40-len(t))
                f.write(t+spa+str(np.round(vals[J+shift],3))+'\n')
        f.close()
    
    if verbose:
        with open(file_name, 'r') as f:
            for i, line in enumerate(f):
                print(line.strip())
                if i == 23:
                    print('-' * 50)
                    print('To see all the results please refer to this file:')
                    print(file_name)
                    break

#================================================================================================#
# Get patches of 2d SEM images
def Patch2DImg(ImgDir, outputDir, res_surface, res_cross, windowsSize=(400,400), m=4, n=2, createDir=True):
    #1.Ensure output dir exists
    os.makedirs(outputDir, exist_ok = True)
    patch_surface = []
    patch_cross = []
    #if os.path.isdir(ImgDir):
    for file in os.listdir(ImgDir):
        if file.lower().endswith(".png"):
            image_path=os.path.join(ImgDir, file)
            basename=os.path.splitext(os.path.basename(image_path))[0]

            #Image processing
            Img=cv2.imread(image_path)
            NLM=cv2.fastNlMeansDenoisingColored(Img, None, 10, 10, 7, 21) 
            Img=cv2.cvtColor(NLM, cv2.COLOR_BGR2GRAY) # convert into gray image
            #Get image size
            hei = Img.shape[0]
            wid = Img.shape[1]
            cross_ratio = res_surface/res_cross
            cross_size = tuple(np.ceil((wid*cross_ratio,hei*cross_ratio)).astype(int))
            patch_wid, patch_hei = windowsSize

            mid_lon = np.int32(np.ceil(np.linspace(0,wid - patch_wid,m)))
            mid_lan = np.int32((np.linspace(0,hei - patch_hei,n)))

            start_lon = mid_lon
            end_lon = mid_lon + patch_wid

            if "surface" in basename:
                if int(cross_ratio) != 1:
                    Img = cv2.resize(Img,cross_size,interpolation=cv2.INTER_AREA)
                    hei = Img.shape[0]
                    wid = Img.shape[1]
                    mid_lon = np.int32(np.ceil(np.linspace(0,wid - patch_wid,m)))
                    mid_lan = np.int32(np.ceil(np.linspace(0,hei - patch_hei,n)))
                    start_lon = mid_lon
                    end_lon = mid_lon + patch_wid
                patch_count = 0
                start_lan = mid_lan
                end_lan = mid_lan + patch_hei
                for x in range(m):
                    for y in range(n):
                        patch = Img[start_lan[y]:end_lan[y], start_lon[x]:end_lon[x]]
                        patch = cv2.resize(patch,(ImgSize,ImgSize),interpolation=cv2.INTER_AREA)
                        patch_count += 1
                        ImgPath = f'{outputDir}/Surface_Patch_#{patch_count}.png'
                        if createDir == True:
                            cv2.imwrite(ImgPath, patch)
                        patch_surface.append(patch)
            elif "cross_section" in basename:
#                 if int(cross_ratio) != 1:
#                     Img = cv2.resize(Img,cross_size,interpolation=cv2.INTER_LINEAR)
                patch_count = 0
                mid_lan = np.ceil(hei/2)
                start_lan = np.int32(mid_lan - patch_hei / 2)
                end_lan = np.int32(mid_lan + patch_hei / 2)
                for x in range(m):
                        patch = Img[start_lan:end_lan, start_lon[x]:end_lon[x]]
                        patch = cv2.resize(patch,(ImgSize,ImgSize),interpolation=cv2.INTER_AREA)
                        patch_count += 1
                        ImgPath = f'{outputDir}/Cross_section_Patch_#{patch_count}.png'
                        if createDir == True:   
                            cv2.imwrite(ImgPath, patch)
                        patch_cross.append(patch)
            else:
                print("Can't find any image!")
    return patch_surface, patch_cross


#================================================================================================#
# Other functions
#================================================================================================#