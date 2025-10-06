#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Input,MaxPooling2D,BatchNormalization,Flatten,Dense,concatenate,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16,VGG19,ResNet101,InceptionV3,EfficientNetB0
from tensorflow.keras.optimizers import Adam,RMSprop

Features_Num = int(308)
#================================================================================================#
# Neural network architecture
#================================================================================================#

# Network #1 (Fig.4 in the paper)
def FiltNet3(INPUT_SHAPE, INPUT_SHAPE_2, OUTPUT_SHAPE):
    # fixed filter size/ 3 convs
    inputs = Input(INPUT_SHAPE[1:])
    inputs_2 = Input(INPUT_SHAPE_2[1:]) # Assuming single metadata feature
   
    c1 = Conv2D(12, (3, 3), kernel_initializer='he_normal',
                padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(24, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(36, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
    p3 = MaxPooling2D((2, 2))(c3)
    f = Flatten()(p3)
    d1 = Dense(Features_Num, activation=tf.nn.relu)(f)
    d2 = Dense(Features_Num, activation=tf.nn.relu)(d1)
    
    merged = concatenate([d2, inputs_2])
    
    d3 = Dense(Features_Num, activation="relu")(merged)
    x3 = Dropout(0.5)(d3) # Regularization technique for preventing overfitting
    
    # Output layer
    outputs = Dense(Features_Num, activation="relu")(x3)
    model = Model(inputs=[inputs, inputs_2], outputs=[outputs])
    optim = tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer='adam',  loss='mse', metrics=['mse'])
    return model

# Network #2 (Fig.4 in the paper)
def FiltNet24(INPUT_SHAPE, INPUT_SHAPE_2, OUTPUT_SHAPE):
    # VGG16,2dense + 1 dense + 1dense;adam optimizer; loss: MSE
    inputs = Input(INPUT_SHAPE[1:])
    inputs_2 = Input(INPUT_SHAPE_2[1:]) # Assuming single metadata feature
    base_model = VGG16(weights="imagenet", 
                       include_top=False, 
                       input_tensor=inputs,
                       input_shape=(256,256,3)
                       )
    base_model.trainable = False
    #c1 = base_model(inputs, training=False)
    c1 = base_model.output
    f = Flatten()(c1)
    d1 = Dense(Features_Num, activation=tf.nn.relu)(f)
    d2 = Dense(Features_Num, activation=tf.nn.relu)(d1)
   # Concatenate layer
    merged = concatenate([d2, inputs_2])
    
    # Dense layers 
    d3 = Dense(Features_Num, activation="relu")(merged)
    x3 = Dropout(0.5)(d3) # Regularization technique for preventing overfitting
    
    # Output layer
    outputs = Dense(Features_Num, activation="relu")(x3)
    model = Model(inputs=[inputs, inputs_2], outputs=[outputs])
    optim = tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer='adam',  loss='mse', metrics=['mse'])
    return model

# Network #3 (Fig.4 in the paper)
def FiltNet25(INPUT_SHAPE, INPUT_SHAPE_2, OUTPUT_SHAPE):
    # VGG19,2dense + 1 dense + 1dense; adam optimizer; loss: MSE
    inputs = Input(INPUT_SHAPE[1:])
    inputs_2 = Input(INPUT_SHAPE_2[1:]) # Assuming single metadata feature
    base_model = VGG19(weights="imagenet", 
                       include_top=False, 
                       input_tensor=inputs,
                       input_shape=(256,256,3)
                       )
    base_model.trainable = False
    c1 = base_model.output
    f = Flatten()(c1)
    d1 = Dense(Features_Num, activation=tf.nn.relu)(f)
    d2 = Dense(Features_Num, activation=tf.nn.relu)(d1)
   # Concatenate layer
    merged = concatenate([d2, inputs_2])
    
    # Dense layers 
    d3 = Dense(Features_Num, activation="relu")(merged)
    x3 = Dropout(0.5)(d3) # Regularization technique for preventing overfitting
    
    # Output layer
    outputs = Dense(Features_Num, activation="relu")(x3)
    model = Model(inputs=[inputs, inputs_2], outputs=[outputs])
    optim = tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer='adam',  loss='mse', metrics=['mse'])
    return model

# Network #4 (Fig.4 in the paper)
def FiltNet26(INPUT_SHAPE, INPUT_SHAPE_2, OUTPUT_SHAPE):
    # ResNet101; 2dense + 1 dense + 1dense;; adam optimizer; loss: mse
    inputs = Input(INPUT_SHAPE[1:])
    inputs_2 = Input(INPUT_SHAPE_2[1:]) # Assuming single metadata feature
    base_model = ResNet101(weights="imagenet", 
                       include_top=False, 
                       input_tensor=inputs,
                       input_shape=(256,256,3)
                       )
    base_model.trainable = False
    c1 = base_model.output
    f = Flatten()(c1)
    d1 = Dense(Features_Num, activation=tf.nn.relu)(f)
    d2 = Dense(Features_Num, activation=tf.nn.relu)(d1)
    
   # Concatenate layer
    merged = concatenate([d2, inputs_2])
    
    # Dense layers 
    d3 = Dense(Features_Num, activation="relu")(merged)
    x3 = Dropout(0.5)(d3) # Regularization technique for preventing overfitting
    
    # Output layer
    outputs = Dense(Features_Num, activation="relu")(x3)
    model = Model(inputs=[inputs, inputs_2], outputs=[outputs])
    optim = tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer='adam',  loss='mse', metrics=['mse'])
    return model

# Network #5 (Fig.4 in the paper)
def FiltNet27(INPUT_SHAPE, INPUT_SHAPE_2, OUTPUT_SHAPE):
    # InceptionV3; 1dense + 2 dense + 1dense; adam optimizer; loss: mse
    inputs = Input(INPUT_SHAPE[1:])
    inputs_2 = Input(INPUT_SHAPE_2[1:]) # Assuming single metadata feature
    base_model = InceptionV3(weights="imagenet", 
                       include_top=False, 
                       input_tensor=inputs,
                       input_shape=(256,256,3)
                       )
    base_model.trainable = False
    c1 = base_model.output
    f = Flatten()(c1)
    d1 = Dense(Features_Num, activation=tf.nn.relu)(f)
    d2 = Dense(Features_Num, activation=tf.nn.relu)(d1)
   # Concatenate layer
    merged = concatenate([d2, inputs_2])
    
    # Dense layers 
    d3 = Dense(Features_Num, activation="relu")(merged)
    x3 = Dropout(0.5)(d3) # Regularization technique for preventing overfitting
    
    # Output layer
    outputs = Dense(Features_Num, activation="relu")(x3)
    model = Model(inputs=[inputs, inputs_2], outputs=[outputs])
    optim = tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer='adam',  loss='mse', metrics=['mse'])
    return model

# Network #6 (Fig.4 in the paper)
def FiltNet28(INPUT_SHAPE, INPUT_SHAPE_2, OUTPUT_SHAPE):
    # InceptionV3;  2dense (2drop) + 1 dense + 1dense; adam optimizer; loss: mse
    inputs = Input(INPUT_SHAPE[1:])
    inputs_2 = Input(INPUT_SHAPE_2[1:]) # Assuming single metadata feature
    base_model = InceptionV3(weights="imagenet", 
                       include_top=False, 
                       input_tensor=inputs,
                       input_shape=(256,256,3)
                       )
    base_model.trainable = False
    c1 = base_model.output
    f = Flatten()(c1)
    d1 = Dense(Features_Num, activation=tf.nn.relu)(f)
    x1 = Dropout(0.5)(d1) # Regularization technique for preventing overfittingx1
    d2 = Dense(Features_Num, activation=tf.nn.relu)(x1)
    x2 = Dropout(0.5)(d2) # Regularization technique for preventing overfitting 
   # Concatenate layer
    merged = concatenate([x2, inputs_2])
    
    # Dense layers 
    d3 = Dense(Features_Num, activation="relu")(merged)
    x3 = Dropout(0.5)(d3) # Regularization technique for preventing overfitting
    
    # Output layer
    outputs = Dense(Features_Num, activation="relu")(x3)
    model = Model(inputs=[inputs, inputs_2], outputs=[outputs])
    optim = tf.keras.optimizers.RMSprop(1e-5)
    model.compile(optimizer='adam',  loss='mse', metrics=['mse'])
    return model

# Network #7 (Fig.4 in the paper)
def FiltNet38(INPUT_SHAPE, INPUT_SHAPE_2, OUTPUT_SHAPE):
    # fixed filters / 3 convs;  1dense + 2 dense + 1dense; adam; loss: MSE(0.5) + MSE(1.0)
    inputs = Input(INPUT_SHAPE[1:])
    inputs_2 = Input(INPUT_SHAPE_2[1:]) # Assuming single metadata feature
    base_model = VGG16(weights="imagenet", 
                       include_top=False, 
                       input_tensor=inputs,
                       input_shape=(256,256,3)
                       )
    base_model.trainable = False
    #c1 = base_model(inputs, training=False)
    c1 = base_model.output
    f = tf.keras.layers.Flatten()(c1)
    d1 = tf.keras.layers.Dense(Features_Num, activation=tf.nn.relu)(f)
    d2 = tf.keras.layers.Dense(Features_Num, activation=tf.nn.relu)(d1)
    #d2 = tf.keras.layers.Dense(Features_Num, activation=tf.nn.sigmoid)(d1)
    
   # Concatenate layer
    merged = tf.keras.layers.concatenate([d2, inputs_2])
    
    # Dense layers 
    d3 = tf.keras.layers.Dense(Features_Num, activation="relu")(merged)
    x3 = tf.keras.layers.Dropout(0.5)(d3) # Regularization technique for preventing overfitting
    
    # Output layer
    # Separate output layer for orientation tensor (3 values)
    orientation_output = tf.keras.layers.Dense(3, activation="relu", name="orientation_output")(x3)
    FFE_output = tf.keras.layers.Dense(19, activation="sigmoid", name="FFE_output")(x3)

    # Separate output layer for other outputs (305 values)
    other_output = tf.keras.layers.Dense(286, activation="relu", name="other_output")(x3)
    
    model = Model(inputs=[inputs, inputs_2], outputs=[orientation_output, FFE_output,other_output])

    # Compile the model with custom losses and weights for each output
    model.compile(optimizer="adam", 
                  loss={'orientation_output': 'mse','FFE_output': 'mse', 'other_output': 'mse'},
                  loss_weights={'orientation_output': 1, 'FFE_output': 0.1, 'other_output': 0.1},
                  metrics={'orientation_output': 'mse', 'FFE_output': 'mse','other_output': 'mse'})
    return model

# Network #8 (Fig.4 in the paper)
def FiltNet40(INPUT_SHAPE, INPUT_SHAPE_2, OUTPUT_SHAPE):
    # fixed filters / 3 convs;  1dense + 2 dense + 1dense; adam; loss: MSE(0.5) + MSE(1.0)
    inputs = Input(INPUT_SHAPE[1:])
    inputs_2 = Input(INPUT_SHAPE_2[1:]) # Assuming single metadata feature
    base_model = VGG16(weights="imagenet", 
                       include_top=False, 
                       input_tensor=inputs,
                       input_shape=(256,256,3)
                       )
    base_model.trainable = False
    #c1 = base_model(inputs, training=False)
    c1 = base_model.output
    f = tf.keras.layers.Flatten()(c1)
    d1 = tf.keras.layers.Dense(Features_Num, activation=tf.nn.relu)(f)
    d2 = tf.keras.layers.Dense(Features_Num, activation=tf.nn.relu)(d1)
    #d2 = tf.keras.layers.Dense(Features_Num, activation=tf.nn.sigmoid)(d1)
    
   # Concatenate layer
    merged = tf.keras.layers.concatenate([d2, inputs_2])
    
    # Dense layers 
    d3 = tf.keras.layers.Dense(Features_Num, activation="relu")(merged)
    x3 = tf.keras.layers.Dropout(0.5)(d3) # Regularization technique for preventing overfitting
    
    # Output layer
    # Separate output layer for orientation tensor (3 values)
    orientation_output = tf.keras.layers.Dense(3, activation="relu", name="orientation_output")(x3)
    FFE_output = tf.keras.layers.Dense(19, activation="sigmoid", name="FFE_output")(x3)

    # Separate output layer for other outputs (305 values)
    other_output = tf.keras.layers.Dense(286, activation="relu", name="other_output")(x3)
    
    model = Model(inputs=[inputs, inputs_2], outputs=[orientation_output, FFE_output,other_output])

    # Compile the model with custom losses and weights for each output
    model.compile(optimizer="adam", 
                  loss={'orientation_output': 'mse','FFE_output': 'mse', 'other_output': 'mse'},
                  loss_weights={'orientation_output': 0.5, 'FFE_output': 1.0, 'other_output': 1.0},
                  metrics={'orientation_output': 'mse', 'FFE_output': 'mse','other_output': 'mse'})
    return model

# Network #9 (Fig.4 in the paper)
def FiltNet42(INPUT_SHAPE, INPUT_SHAPE_2, OUTPUT_SHAPE):
    # fixed filters / 3 convs;  1dense + 2 dense + 1dense; adam; loss: MSE(0.5) + MSE(1.0)
    inputs = Input(INPUT_SHAPE[1:])
    inputs_2 = Input(INPUT_SHAPE_2[1:]) # Assuming single metadata feature
    base_model = VGG16(weights="imagenet", 
                       include_top=False, 
                       input_tensor=inputs,
                       input_shape=(256,256,3)
                       )
    base_model.trainable = False
    #c1 = base_model(inputs, training=False)
    c1 = base_model.output
    f = tf.keras.layers.Flatten()(c1)
    d1 = tf.keras.layers.Dense(Features_Num, activation=tf.nn.relu)(f)
    d2 = tf.keras.layers.Dense(Features_Num, activation=tf.nn.relu)(d1)
    #d2 = tf.keras.layers.Dense(Features_Num, activation=tf.nn.sigmoid)(d1)
    
   # Concatenate layer
    merged = tf.keras.layers.concatenate([d2, inputs_2])
    
    # Dense layers 
    d3 = tf.keras.layers.Dense(Features_Num, activation="relu")(merged)
    x3 = tf.keras.layers.Dropout(0.5)(d3) # Regularization technique for preventing overfitting
    
    # Output layer
    # Separate output layer for orientation tensor (3 values)
    orientation_output = tf.keras.layers.Dense(3, activation="relu", name="orientation_output")(x3)
    FFE_output = tf.keras.layers.Dense(19, activation="sigmoid", name="FFE_output")(x3)

    # Separate output layer for other outputs (305 values)
    other_output = tf.keras.layers.Dense(286, activation="relu", name="other_output")(x3)
    
    model = Model(inputs=[inputs, inputs_2], outputs=[orientation_output, FFE_output,other_output])

    # Compile the model with custom losses and weights for each output
    model.compile(optimizer="adam", 
                  loss={'orientation_output': 'mse','FFE_output': 'mse', 'other_output': 'mse'},
                  loss_weights={'orientation_output': 1.0, 'FFE_output': 1.0, 'other_output': 1.0},
                  metrics={'orientation_output': 'mse', 'FFE_output': 'mse','other_output': 'mse'})
    return model

def modelmake(INPUT_SHAPE,INPUT_SHAPE_2,OUTPUT_SHAPE,ModelType):
    model_classes = {
        3: FiltNet3,
        24: FiltNet24,
        25: FiltNet25,
        26: FiltNet26,
        27: FiltNet27,
        28: FiltNet28,  
        38: FiltNet38,
        40: FiltNet40,
        42: FiltNet42,
    }
    model_class = model_classes.get(ModelType)
    if model_class is None:
        raise ValueError(f"ModelType {ModelType} is not recognized.")
    model = model_class(INPUT_SHAPE, INPUT_SHAPE_2, OUTPUT_SHAPE)
    
    return model  