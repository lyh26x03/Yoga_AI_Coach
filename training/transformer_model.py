import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, LayerNormalization, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2

def transformer_encoder(inputs, num_heads, key_dim, ff_dim, dropout_rate):
    """Transformer Encoder Layer"""
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)  # Residual connection

    # Feed Forward Network (FFN)
    ff_output = Dense(ff_dim, activation='relu', kernel_regularizer=l2(0.0005))(attention_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = LayerNormalization(epsilon=1e-6)(attention_output + ff_output)  # Residual connection

    return ff_output

def create_transformer_model(input_shape, num_classes, num_layers, num_heads, key_dim, ff_dim, dropout_rate):
    """Build Transformer Model"""
    inputs = Input(shape=input_shape)

    # Positional Encoding
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_embedding_layer = Embedding(input_dim=input_shape[0], output_dim=input_shape[1])
    pos_encoding = pos_embedding_layer(positions)
    pos_encoding = tf.expand_dims(pos_encoding, axis=0)  # Shape (1, 30, 28)

    x = inputs + pos_encoding  # Add Positional Encoding

    # Multiple Transformer Encoder layers
    for _ in range(num_layers):
        x = transformer_encoder(x, num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim, dropout_rate=dropout_rate)

    # Pooling Layer
    x = GlobalAveragePooling1D()(x)

    # Output Layer
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.001))(x)

    model = Model(inputs, outputs)
    return model