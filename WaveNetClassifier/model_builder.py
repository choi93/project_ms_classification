from tensorflow.keras.layers import Input, Reshape, Activation, Add, Conv1D, AveragePooling1D
from tensorflow.keras.models import Model

class ModelBuilder:
    @staticmethod
    def add_conv_layer(tensor, filters, name, kernel_size=100, activation=None):
        return Conv1D(
            filters,
            kernel_size,
            padding='same',
            activation=activation,
            name=name
        )(tensor)

    @staticmethod
    def build_model(config, residual_blocks):
        x_input = Input(shape=(config.data_params.input_shape,), name='original_input')
        x = Reshape((config.data_params.input_shape,) + (1,), name='reshaped_input')(x_input)

        skip_connections = []
        x = Conv1D(
            config.model_params.n_filters,
            config.model_params.kernel_size,
            dilation_rate=1,
            padding='causal',
            name='dilated_conv_1'
        )(x)

        for block in residual_blocks:
            x, skip = block.build(x)
            skip_connections.append(skip)

        out = Add(name='skip_connections')(skip_connections)
        out = Activation('relu')(out)
        
        # Downsampling layers
        out = ModelBuilder.add_conv_layer(out, 80, 'conv_5ms', activation='relu')
        out = AveragePooling1D(80, padding='same', name='downsample_to_200Hz')(out)

        out = ModelBuilder.add_conv_layer(out, 100, 'conv_500ms', activation='relu')
        out = ModelBuilder.add_conv_layer(out, config.data_params.output_shape, 'conv_500ms_target_shape', activation='relu')
        out = AveragePooling1D(100, padding='same', name='downsample_to_2Hz')(out)

        final_conv_size = config.data_params.input_shape // 8000
        out = ModelBuilder.add_conv_layer(out, config.data_params.output_shape, 'final_conv', kernel_size=final_conv_size, activation=None)
        out = AveragePooling1D(final_conv_size, name='final_pooling')(out)

        out = Reshape((config.data_params.output_shape,))(out)
        out = Activation('softmax')(out)

        return Model(x_input, out) 