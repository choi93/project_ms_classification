from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History, ModelCheckpoint, Callback
from tensorflow_addons.metrics import FBetaScore

from .residual_block import ResidualBlock
from .model_builder import ModelBuilder
from .config import Config, DataParams, ModelParams, TrainParams

class WaveNetClassifier(Model):
    def __init__(self, config_path: str = 'input_train.yaml'):
        super(WaveNetClassifier, self).__init__()
        self.config = self.load_config(config_path)
        self.residual_blocks = self.create_residual_blocks()
        self.model = ModelBuilder.build_model(self.config, self.residual_blocks)

    def load_config(self, config_path: str) -> Config:
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        config = Config(
            data_params=DataParams(**config_dict.get('data_params', {})),
            model_params=ModelParams(**config_dict.get('model_params', {})),
            train_params=TrainParams(**config_dict.get('train_params', {}))
        )
        return config

    def create_residual_blocks(self) -> List[ResidualBlock]:
        return [
            ResidualBlock(
                n_filters=self.config.model_params.n_filters,
                kernel_size=self.config.model_params.kernel_size,
                dilation_rate=self.config.model_params.kernel_size ** i,
                index=i
            ) for i in range(1, self.config.model_params.dilation_depth + 1)
        ]

    def compile_model(self):
        metric_list = [
            'accuracy',
            FBetaScore(num_classes=self.config.data_params.output_shape, beta=self.config.train_params.beta, average=None)
        ]
        self.model.compile(optimizer=self.config.train_params.optimizer, loss=self.config.train_params.loss, metrics=metric_list)

    def _create_callbacks(self, validation_data: bool, validation_split: Optional[float]) -> List[Callback]:
        """학습에 사용될 콜백들을 생성합니다."""
        callbacks = []
        if not self.config.train_params.save:
            return callbacks, None

        checkpoint_path = self.config.train_params.save_dir + "wave_clas-{epoch:02d}.h5"
        history_path = f"{self.config.train_params.save_dir}wavenet_classifier_training_history.csv"
        monitor = 'val_accuracy' if validation_data or validation_split else 'accuracy'
        
        checkpointer = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            verbose=1,
            save_best_only=False
        )
        history = History()
        callbacks.extend([history, checkpointer])
        return callbacks, history_path

    def fit_model(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        validation_split: Optional[float] = None,
        sample_weight: Optional[np.ndarray] = None
    ):
        
        callbacks, history_path = self._create_callbacks(bool(val_data), validation_split)

        training_history = self.model.fit(
            train_data,
            train_labels,
            shuffle=True,
            batch_size=self.config.train_params.batch_size,
            epochs=self.config.train_params.epochs,
            validation_split=validation_split,
            validation_data=val_data,
            callbacks=callbacks,
            sample_weight=sample_weight
        )

        if self.config.train_params.save and callbacks:
            df = pd.DataFrame.from_dict(training_history.history)
            df.to_csv(history_path, encoding='utf-8', index=False)
            
        return training_history

    def call(self, inputs):
        return self.model(inputs)

    def predict_classes(self, x: np.ndarray) -> np.ndarray:
        """
        입력 데이터에 대한 클래스 예측을 수행합니다.
        
        Args:
            x: 예측할 입력 데이터
        
        Returns:
            예측된 클래스 레이블
        """
        predictions = self.model.predict(x)
        return np.argmax(predictions, axis=1)
