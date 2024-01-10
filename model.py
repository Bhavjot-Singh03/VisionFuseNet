K.clear_session()
from tensorflow.keras.applications import EfficientNetV2B3

## Transformer Mechanism 

def MLP(x, mlp_dim, dim, dropout_rate = 0.2):

    x = Dense(mlp_dim, activation = 'swish')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dim)(x)
    x = Dropout(dropout_rate)(x)

    return x
  
def Trans_Encoder(inputs, num_heads, hidden_dim, mlp_dim):
    
    skip_1 = inputs
    x = LayerNormalization()(inputs)
    x = MultiHeadAttention(num_heads = num_heads, key_dim = hidden_dim)(x, x)
    x = Add()([skip_1, x])

    skip_2 = x
    x = LayerNormalization()(x)
    x = MLP(x, mlp_dim, hidden_dim)
    x = Add()([skip_2, x])

    return x

def MHA_RESIDUAL_CONV(inputs, dilation_rate = 1, filters = 256):
    
    B, H, W, C = inputs.shape
    skip = inputs
    
    x = Conv2D(filters = filters, kernel_size = (1,1), dilation_rate = dilation_rate, padding = 'same', use_bias = False)(inputs)
    x = BatchNormalization()(x)

    """Patch Embeddings"""
    patch_embed = Activation('swish')(x)
    _, h, w, f = patch_embed.shape
    patch_embed = Reshape((h*w, f))(patch_embed) 
    
    if H == 16 :
        """Positional Embedding -> Number of patches : 128*128/(patch_size * patch_size) = N"""
        patch_size = 8
        N = (128*128)//(patch_size * patch_size) 
        positions = tf.range(start = 0, limit = N, delta = 1) # (256, )
        pos_embed = Embedding(input_dim = N, output_dim = filters)(positions) # (256, 256)
        embedding = patch_embed + pos_embed
        x = embedding
        
    if H == 8 :
        patch_size = 16
        N = (128*128)//(patch_size * patch_size) 
        positions = tf.range(start = 0, limit = N, delta = 1) # (64, )
        pos_embed = Embedding(input_dim = N, output_dim = filters)(positions) # (64, 256)
        embedding = patch_embed + pos_embed
        x = embedding   
        
    if H == 4 :
        patch_size = 32
        N = (128*128)//(patch_size * patch_size) 
        positions = tf.range(start = 0, limit = N, delta = 1) # (16, )
        pos_embed = Embedding(input_dim = N, output_dim = filters)(positions) # (16, 256)
        embedding = patch_embed + pos_embed
        x = embedding

    T = Trans_Encoder(x, 3, filters, filters * 2)
    T = LayerNormalization()(T)
    T = Reshape((H,W,filters))(T)
   
    skip = Conv2D(filters = filters, kernel_size = (1,1), dilation_rate = dilation_rate, padding = 'same', use_bias = False)(skip)
    skip = Add()([T, skip])
    skip = BatchNormalization()(skip)
    skip = Activation('swish')(skip)
        
    return skip

## U-Net mechanism

def Trans_Unet(features):

    e1 = MHA_RESIDUAL_CONV(features, dilation_rate = 6, filters = 256)
    p1 = MaxPooling2D(pool_size = (2,2))(e1)
    
    e2 = MHA_RESIDUAL_CONV(p1, dilation_rate = 12, filters = 256)
    p2 = MaxPooling2D(pool_size = (2,2))(e2)
    
    e3 = MHA_RESIDUAL_CONV(p2, dilation_rate = 18, filters = 256)
    
    u1 = Conv2DTranspose(filters = 256, kernel_size = (2,2), strides = (2,2))(e3)
    c1 = Concatenate()([u1, e2])
    c1 = MHA_RESIDUAL_CONV(c1, dilation_rate = 18, filters = 256)
    
    u2 = Conv2DTranspose(filters = 256, kernel_size = (2,2), strides = (2,2))(c1)
    c2 = Concatenate()([u2, e1])
    c2 = MHA_RESIDUAL_CONV(c2, dilation_rate = 12, filters = 256)
    
    return c2

## DeepLab

def DeepLabV3(shape, output_channels, output_activation):
    inputs = Input(shape)

    """-------------------ENCODER------------------------"""
    
    """Using EfficientNetV2B3 as a base_model"""
    base_model = EfficientNetV2B3(weights = 'imagenet', include_top = False, input_tensor=inputs)
    features = base_model.get_layer('block6a_expand_activation').output 

    """Pooling"""
    shape = features.shape
    
    a = AveragePooling2D(pool_size = (shape[1], shape[2]))(features) 
    a = Conv2D(filters = 256, kernel_size = (1,1), padding = 'same', use_bias = False)(a)
    a = BatchNormalization()(a)
    a = Activation('swish')(a)
    a = UpSampling2D(size = (shape[1], shape[2]), interpolation = 'bilinear')(a) 
    
    """Trans_Unet"""
    e = Trans_Unet(features)
    
    """Concatenation"""
    e_out = Concatenate()([e, a])
    
    """MHSAR"""
    e_out = MHA_RESIDUAL_CONV(e_out, dilation_rate = 1, filters = 256)
    
    # Upsampling by 2
    e_a = Conv2DTranspose(filters = 64, kernel_size = (2,2), strides = (2,2))(e_out) 
    
    """-------------------DECODER------------------------"""

    """Extracting Low level feature from EfficientNetV2B3""" 
    d_l = base_model.get_layer('block4a_expand_activation').output 
    d_l = Conv2D(filters = 64, kernel_size = (1,1), padding = 'same', use_bias = False)(d_l)
    d_l = BatchNormalization()(d_l)
    d_l = Activation('swish')(d_l)
    d_l = Dropout(0.2)(d_l)

    ed = Concatenate()([e_a, d_l]) 
  
    d = Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', use_bias = False)(ed)
    d = BatchNormalization()(d)
    d = Activation('swish')(d)
    d = Dropout(0.2)(d)
    
    d = Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', use_bias = False)(d)
    d = BatchNormalization()(d)
    d = Activation('swish')(d)
    d = Dropout(0.2)(d)

    d = Conv2DTranspose(filters = 256, kernel_size = (8,8), strides = (8,8))(d) 

    output = Conv2D(filters = output_channels, kernel_size = (1,1), activation = output_activation)(d)
    print(output.shape)
  
    return Model(inputs, output, name = 'MHA_DEEPLABV3_PLUS')

## Instantiating the Model

K.clear_session()
output_channels = 1
activation = 'sigmoid'
shape = (256, 256, 3)

MHA_DEEPLAB = DeepLabV3(shape, output_channels, activation)

total_params = 0
for layer in MHA_DEEPLAB.trainable_weights:
    total_params += tf.keras.backend.count_params(layer)

print(f"Total number of parameters: {total_params}")


