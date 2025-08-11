# 분류 모델 성능 분석

CIFAR10 데이터셋에 대한 분류 모델 학습 프로젝트

---

## 🚀 개요

주어진 데이터를 바탕으로 특정 결과를 예측하는 딥러닝 기반의 분류 모델을 개발하는 것을 목표로 했습니다. 학습 과정 내내 정확도는 꾸준히 향상되었고, 손실 값은 안정적으로 감소하는 모습을 보여 성공적으로 모델을 학습시킬 수 있었습니다.

---

## 📋 학습 환경 및 설정

- 프레임워크: PyTorch
- 최적화 도구 (Optimizer): Adam
- 손실 함수 (Loss Function): CrossEntropyLoss
- 학습률 (Learning Rate): 0.001
- 에포크 (Epochs): 10
- 평가 지표: 정확도 (Accuracy), 손실 (Loss)
- 데이터셋: 학습(Training) 데이터와 검증(Validation) 데이터 분리

---

## 📈 학습 결과

### 학습 결과 표
| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
|-------|------------|----------------|-----------------|---------------------|
| 1     | 1.4599     | 46.89%         | 1.1647          | 58.52%              |
| 2     | 1.0813     | 61.24%         | 0.9322          | 66.82%              |
| 3     | 0.9605     | 65.93%         | 0.9551          | 66.04%              |
| ...   | ...        | ...            | ...             | ...                 |
| 10    | 0.6959     | 76.01%         | 0.6126          | 78.46%              |

### Accuracy and Loss Trends
#### Accuracy
- 학습 정확도는 Epoch마다 상승하여 최종 Epoch에선 **76.01%**를 달성했습니다.
- 검증 정확도 또한 비슷한 추세로, **78.46%**에 도달했습니다.
- 모델이 특정 데이터에 과적합되지 않고, 새로운 데이터에도 잘 일반화될 수 있음 확인할 수 있었습니다.

#### Loss
- 학습 손실은 지속적으로 감소하며 0.6959까지 낮아졌습니다.
- 검증 손실도 마찬가지로 감소하여 0.6126으로 낮아졌습니다.
- 모델이 안정적으로 학습을 확인할 수 있었습니다.

#### Training VS Validation
![Plot](./outputs/output.png)

---

## 🔍 Analysis

### Key Observations
1. **Performance**:
   - Steady improvement in training and validation accuracy across epochs.
   - Validation accuracy closely follows training accuracy, showing no signs of overfitting.
2. **Efficiency**:
   - Model performance plateaued by epoch 10, suggesting the training process was efficient and sufficient.

### Strengths
- High generalization capability.
- Stable and reliable learning process.

### Areas for Improvement
- Explore further hyperparameter tuning.
- Add data augmentation to enhance model robustness.

---

## 🛠 How to Use

### 1. Prepare the Environment
Ensure you have the required dependencies installed. Use the following command to set up:

```bash
pip install -r requirements.txt
```

### 2. Run the Training Script
Train the model using the provided script:

```bash
python train.py
```

### 3. Visualize Results
Generated graphs and logs will be saved in the outputs/ directory.

🌟 Future Work
- Hyperparameter Optimization: Fine-tune the learning rate, batch size, and architecture to further improve performance.
- Data Augmentation: Enhance the dataset using augmentation techniques to boost generalization.
- Testing: Evaluate the model on a separate test dataset to confirm final performance.

📂 Repository Structure

```bash
├── data/              # Dataset
├── outputs/           # Training logs and visualizations
├── train.py           # Training script
├── requirements.txt   # Dependencies
└── README.md          # Project 
```
