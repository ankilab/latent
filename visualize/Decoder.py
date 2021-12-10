
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Input, Conv2D, MaxPooling2D, UpSampling2D, Activation, SeparableConv2D, ReLU, Concatenate

def convblock(x, filters, batch_norm=True, use_bias=False, activation='relu', name='conv_layer'):
    x = Conv2D(filters, 
            (3, 3), 
            use_bias=use_bias, 
            padding="same",
            strides=1,
            name=name,
            kernel_initializer='he_uniform')(x)
    
    if batch_norm:
        x = BatchNormalization()(x)
    
    if activation == 'relu6':
        x = ReLU(6.)(x)

    x = Activation(activation)(x)

    return x
    
def Decoder(input_shape=(32,16,1), 
         decoder_filter=[128, 64, 32, 16],
         activation='sigmoid', 
         classes=1, 
         use_bias=False, 
         conv_activation='relu',
         batch_norm=True):

    latent_in = Input(shape=input_shape, name="input")
    
    x = latent_in

    # Decoding
    for i, fi in enumerate(decoder_filter):
        x = UpSampling2D(size = (2,2))(x)

        x = convblock(x, fi,
                use_bias=use_bias, batch_norm=batch_norm, 
                activation=conv_activation, name=f'decoding_{i}a')

        x = convblock(x, fi, 
            use_bias=use_bias, batch_norm=batch_norm, 
            activation=conv_activation, name=f'decoding_{i}b')
    
    # Final output, 1x1 convolution 
    model_output = Conv2D(classes, 
                          (1, 1), 
                          use_bias=False, 
                          padding="same",
                          activation=activation,
                          strides=1,
                          name='output',
                          kernel_initializer='glorot_uniform')(x)

    return Model(latent_in, model_output)