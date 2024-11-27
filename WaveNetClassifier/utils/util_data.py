import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
class DataLoader:
    def __init__(self, data_dir, train_file, train_labels_file, val_file=None, val_labels_file=None): 
        self.data_dir = data_dir
        self.train_file = train_file
        self.train_labels_file = train_labels_file
        self.val_file = val_file
        self.val_labels_file = val_labels_file

    def load_data(self):
        x_train = np.load(os.path.join(self.data_dir, self.train_file))
        y_train = np.load(os.path.join(self.data_dir, self.train_labels_file))
        
        if self.val_file:
            x_val = np.load(os.path.join(self.data_dir, self.val_file))
            y_val = np.load(os.path.join(self.data_dir, self.val_labels_file))
        else:
            x_val, y_val = None, None
        return x_train, y_train, x_val, y_val
    
def compute_sample_weight(y_train,class_index=4,weight_factor=2):
    # 각 클래스별 샘플 수 계산
    ns = y_train.sum(axis=0)
    ndata, nclass = y_train.shape
        
    # 가중치 계산
    weight = (1/ns) * (ndata)/nclass
    weight[class_index] *= weight_factor  # 5번째 클래스에 대해 2배 가중치
        
    # 각 샘플의 클래스 인덱스 계산
    y_int = y_train.argmax(axis=1)
        
    #벡터화된 방식으로 샘플 가중치 할당
    sample_weight = weight[y_int]
        
    return sample_weight