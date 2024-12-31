# Automatic Microseismic Signal Classification for Mining Safety Monitoring

This repository contains the implementation of the WaveNet-based automatic microseismic signal classification model described in the paper:

> **Automatic Microseismic Signal Classification for Mining Safety Monitoring Using the WaveNet Classifier**  
> *Woochang Choi, Sukjoon Pyun, Dae-Sung Cheon*  
> Published in *Geophysical Prospecting, 2024*  
> DOI: [10.1111/1365-2478.13398](https://doi.org/10.1111/1365-2478.13398)

---

## 📖 **Overview**

This project focuses on developing a deep learning-based classifier for microseismic signals to enhance mining safety monitoring. The classifier utilizes a modified WaveNet architecture to process raw microseismic data and classify signals into five categories:

- **Scaling**
- **Blasting**
- **Drilling**
- **Electric Noise**
- **Microseismic**

The model is designed to handle imbalanced datasets, mitigate information loss, and operate efficiently in real-time safety monitoring applications.

---

## 🚀 **Features**

- **WaveNet-based Architecture**: Optimized for long time-series data classification without preprocessing.
- **Data Augmentation**: Includes techniques to address data imbalance, such as using external data from similar mines.
- **Performance Comparison**: Benchmarked against Random Forest and SampleCNN models.
- **Real-Time Application**: Capable of processing signals faster than their duration, enabling real-time monitoring.

---

## 📂 **Repository Structure**

```
project_ms_classification/
├── WaveNetClassifier/
│ ├── init.py
│ ├── WaveNetClassifier.py
│ ├── residual_block.py
│ ├── model_builder.py
│ ├── config.py
│ └── utils/
├── data/
├── input_train.yaml
├── main.py
└── test.py
```

---

## 📊 **Results**

The WaveNet-based classifier demonstrated superior performance compared to traditional methods:

- **Accuracy**: 99.96% (AUC score)
- **Recall**: High recall for microseismic signals, critical for safety monitoring.
- **Precision**: Outperformed Random Forest and SampleCNN in misclassification rates.

For detailed results, refer to the paper or the `results/` folder.

---


## 🖥 **Usage**

### **1. Train the Model**
To train the WaveNet model on your own dataset:
```bash
python src/train.py --config configs/train_config.json
```

### **2. Evaluate the Model**
To evaluate a trained model:
```bash
python src/evaluate.py --model_path models/wavenet_best.pth --data_path data/test_data.pkl
```

### **3. Visualize Results**
Use the Jupyter notebooks in the `notebooks/` folder for detailed analysis and visualization:
```bash
jupyter notebook notebooks/visualize_results.ipynb
```

---

## 📁 **Data**

Due to data regulations, the raw dataset cannot be shared publicly. However, example datasets and feature data for the Random Forest model are provided in the `data/` folder. For access to the full dataset, please follow the procedures outlined by the Korea Institute of Geoscience and Mineral Resources.

---

## 📜 **Citation**

If you use this code or find it helpful, please cite the paper:

```
@article{choi2024wavenet,
  title={Automatic Microseismic Signal Classification for Mining Safety Monitoring Using the WaveNet Classifier},
  author={Woochang Choi and Sukjoon Pyun and Dae-Sung Cheon},
  journal={Geophysical Prospecting},
  year={2024},
  volume={72},
  pages={315--332},
  doi={10.1111/1365-2478.13398}
}
```

---

## 🙏 **Acknowledgements**

This work was supported by:
- **Inha University Research Grant**
- **KIGAM’s Basic Research Project (22-3115)**, funded by the Ministry of Trade, Industry and Energy.

Special thanks to the Korea Institute of Geoscience and Mineral Resources for providing the data and resources.

---

