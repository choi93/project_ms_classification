from tensorflow.keras.layers import Conv1D, Multiply, Add

class ResidualBlock:
    def __init__(self, n_filters: int, kernel_size: int, dilation_rate: int, index: int):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.index = index

    def build_conv_layer(self, activation: str, name_prefix: str) -> Conv1D:
        return Conv1D(
            self.n_filters,
            self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='causal',
            name=f'{name_prefix}_{activation}_{self.dilation_rate}_{self.index}',
            activation=activation
        )

    def build(self, x):
        tanh_out = self.build_conv_layer('tanh', 'dilated_conv')(x)
        sigm_out = self.build_conv_layer('sigmoid', 'dilated_conv')(x)

        z = Multiply(name=f'gated_activation_{self.index}')([tanh_out, sigm_out])
        skip = Conv1D(self.n_filters, 1, name=f'skip_{self.index}')(z)
        res = Add(name=f'residual_block_{self.index}')([skip, x])
        
        return res, skip 