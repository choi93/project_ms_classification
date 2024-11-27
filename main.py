import os
from WaveNetClassifier import WaveNetClassifier
from WaveNetClassifier.utils import DataLoader, compute_sample_weight

# GPU setting
os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"
os.environ [ "CUDA_VISIBLE_DEVICES" ] = '0'

data_dir='./data'
train_file='d_train.npy'
train_labels_file='l_train.npy'
val_file='d_val.npy'
val_labels_file='l_val.npy'

data_loader = DataLoader(data_dir, train_file, train_labels_file, val_file, val_labels_file)
x_train, y_train, x_val, y_val = data_loader.load_data()

# set sample weight
sample_weight = compute_sample_weight(y_train,class_index=4,weight_factor=2)
print("data prepared, ready to train!")

# WaveNet 모델 생성 및 학습
wnc = WaveNetClassifier('input_train.yaml')
wnc.compile_model()
wnc.fit_model(
    train_data=x_train,
    train_labels=y_train,
    val_data=(x_val, y_val),
    sample_weight=sample_weight
)
