def create_model(input_shape):
    input_layer = Input(shape=input_shape)
    
    # 第一層：雙向 LSTM
    x = Bidirectional(LSTM(Dense_1, return_sequences=True))(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # 第二層：LSTM + Attention
    x = LSTM(Dense_2, return_sequences=True)(x)
    query = Dense(Dense_2)(x)
    key = Dense(Dense_2)(x)
    value = x
    attention_output = Attention(use_scale=True)([query, key, value])
    
    # 融合 LSTM 輸出與 Attention 輸出
    x = Concatenate()([x, attention_output])
    
    # 第三層：TimeDistributed + Dense
    x = TimeDistributed(Dense(32, activation='relu'))(x)
    
    # 第四層：Global Pooling
    x_max = GlobalMaxPooling1D()(x)
    x_avg = GlobalAveragePooling1D()(x)
    x = Concatenate()([x_max, x_avg])
    
    # 第五層：Dense
    x = Dense(Dense_3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # 輸出層
    output_layer = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model