
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


def Unet(encoder_filter=[16,  32, 64, 128, 1],
         decoder_filter=[128, 64, 32, 16],
         concatenation=[0,0,0,0],
         activation='sigmoid', 
         classes=1, 
         input_shape=None, 
         use_bias=False, 
         conv_activation='relu',
         batch_norm=True):
    """
    Building a U-Net [1]_.
    
    
    Parameters
    ----------
    
    encoder_filter : list, optional
        The number of filters in the encoder.
        The last number is the number of filters in the latent space.
        Default is [16,  32, 64, 128, 1].

    decoder_filter : list, optional
        The number of filters in the decoder.
        Default is [128, 64, 32, 16]
        
    concatenation : list, optional
        Decides on which depth concatenation is enabled (1: enabled, 0: disabled).
        Default is [0, 0, 0, 0]
    
    activation : str, optional
        The activation function in the last layer. Default is sigmoid.
        
    classes : int, optional
        The number of classes in the last layer. Default is 1.
        
    input_shape : tuple, optional
        The input shape of the data. We train the network to have arbitraty
        input shapes, default is None. Otherwise, the tuple has to follow
        the following criterion: (X, Y, channels)

    use_bias : bool, optional
        If bias should be fitted in the conv layers. Default is False.

    conv_activation : str, optional
        The activation function in the conv layers. Default is relu.
        
    batch_norm : bool, optional
        If BatchNormalization should be used. Default is True.
        
        
    Returns
    -------
    
    Keras Model
        A Keras Model containing the U-Net structure.
        
        
    References
    ----------
    
    [1] Ronneberger, O., Fischer, P., & Brox, T. (2015, October). 
    U-net: Convolutional networks for biomedical image segmentation. 
    In International Conference on Medical image computing and 
    computer-assisted intervention (pp. 234-241). Springer, Cham.
    
    """
    if input_shape is None:
        input_shape = (None, None, 1)
        
    model_input = Input(shape=input_shape, name="input")
    
    to_concat = []
    
    x = model_input


    # Encoding 
    for i, fi in enumerate(encoder_filter[:-1]):
        x = convblock(x, fi,  
                use_bias=use_bias, batch_norm=batch_norm, 
                activation=conv_activation, name=f'encoding_{i}a')


        x = convblock(x, fi,
            use_bias=use_bias, batch_norm=batch_norm, 
            activation=conv_activation, name=f'encoding_{i}b')

        to_concat.append(x)
        
        x = MaxPooling2D(pool_size=(2, 2))(x)   
        
    x = convblock(x, encoder_filter[-1],  
                use_bias=use_bias, batch_norm=batch_norm, 
                activation='relu6', name=f'latent')

        
    # Decoding
    for i, fi in enumerate(decoder_filter):
        x = UpSampling2D(size = (2,2))(x)

        if concatenation[::-1][i]:
            x = Concatenate()([x, to_concat[::-1][i]])

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

    return Model(model_input, model_output)