from dataclasses import dataclass, field

@dataclass
class DataParams:
    input_shape: int = 25000
    output_shape: int = 5

@dataclass
class ModelParams:
    kernel_size: int = 2
    dilation_depth: int = 9
    n_filters: int = 40

@dataclass
class TrainParams:
    optimizer: str = 'adam'
    loss: str = 'categorical_crossentropy'
    beta: float = 2.0
    batch_size: int = 32
    epochs: int = 100
    save: bool = False
    save_dir: str = './'

@dataclass
class Config:
    data_params: DataParams = field(default_factory=DataParams)
    model_params: ModelParams = field(default_factory=ModelParams)
    train_params: TrainParams = field(default_factory=TrainParams)