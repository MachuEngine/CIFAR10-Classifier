import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# 1. 장치 설정 함수
def get_device() -> torch.device:
    """
    사용 가능한 장치를 반환합니다. GPU가 사용 가능하면 GPU를, 그렇지 않으면 CPU를 반환합니다.
    
    Returns:
        torch.device: 사용 가능한 장치
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'사용 장치: {device}')
    return device

# 2. 데이터 로드 및 전처리 함수
def load_data(batch_size_train: int = 64, 
              batch_size_val: int = 100,
              batch_size_test: int = 100, 
              num_workers: int = 2,
              validation_split: float = 0.1) -> tuple:
    """
    CIFAR-10 데이터셋을 로드하고 전처리합니다.
    
    Args:
    batch_size_train (int, optional): 학습 데이터 배치 크기. Defaults to 64.
    batch_size_val (int, optional): 검증 데이터 배치 크기. Defaults to 100.
    batch_size_test (int, optional): 테스트 데이터 배치 크기. Defaults to 100.
    num_workers (int, optional): 데이터 로드 시 사용할 워커 수. Defaults to 2.
    validation_split (float, optional): 검증 데이터 비율. Defaults to 0.1.
    
    Returns:
        tuple: (trainloader, valloader, testloader, classes)
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 데이터 증강: 수평 뒤집기
        transforms.RandomCrop(32, padding=4),  # 데이터 증강: 랜덤 크롭
        transforms.ToTensor(),  # 텐서로 변환
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),  # 정규화 (평균)
                             std=(0.2023, 0.1994, 0.2010))  # 정규화 (표준편차)
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010))
    ])

    # 전체 훈련 데이터셋 로드
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform_train)
    
    # 훈련 데이터와 검증 데이터로 분할
    train_indices, val_indices = train_test_split(
        list(range(len(full_trainset))), # full_trainset에 대한 전체 인덱스 리스트 생성
        test_size=validation_split, # 전체 훈련셋의 10% 비율로 검증 데이터셋을 분리하겠다 
        random_state=42, # 데이터 분할 시 랜덤성을 막기 위해 42라는 임의의 숫자를 지정하여 매번 같은 분할 결과를 얻음 
        stratify=full_trainset.targets # 데이터 분할 과정에서 층화 추출(stratified sampling)을 수행하여 각 클래스의 비율이 훈련 세트와 검증 세트 모두에서 동일하게 유지되도록 하는 역할
    )

    # Subset을 사용하여 훈련 세트와 검증 세트 생성
    train_subset = Subset(full_trainset, train_indices)
    val_subset = Subset(full_trainset, val_indices)

    # DataLoader 생성 ---
    trainloader = DataLoader(train_subset, batch_size=batch_size_train,
                             shuffle=True, num_workers=num_workers)
    valloader = DataLoader(val_subset, batch_size=batch_size_val,
                           shuffle=False, num_workers=num_workers)

    # 테스트 데이터셋
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    # DataLoader 생성 ---
    testloader = DataLoader(testset, batch_size=batch_size_test, 
                            shuffle=False, num_workers=num_workers)
    
    # 클래스 이름 정의
    classes = ['airplane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    return trainloader, valloader, testloader, classes

# 3. 데이터 시각화 함수
def visualize_data(trainloader: DataLoader, classes: list):
    """
    학습 데이터셋의 일부 샘플을 시각화합니다.
    
    Args:
        trainloader (DataLoader): 학습 데이터 로더
        classes (list): 클래스 이름 리스트
    """
    def imshow(img):
        img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)  # 정규화 해제
        npimg = img.numpy() # matplotlib 사용을 위해 tensor를 numpy 배열로 변환 
        plt.imshow(np.transpose(npimg, (1, 2, 0))) # # 변환한 numpy 배열의 C,H,W 순서를 H,W,C로 변경
        # PyTorch 텐서는 채널 우선(Channel-first) 형식인 (C,H,W)를 사용합니다. 
        # Matplotlib는 채널 마지막(Channel-last) 형식인 (H,W,C)를 사용합니다.
        plt.show() # imshow로 렌더링한 이미지를 화면에 출력합니다. 
    
    # 데이터 샘플 확인
    dataiter = iter(trainloader) # trainloader를 순회할 iterator 객체를 생성합니다. 
    images, labels = next(dataiter) # ieterator 객체로 부터 images, labels를 하나씩 순회하며 저장합니다. 

    
    imshow(torchvision.utils.make_grid(images[:16])) # 가져온 images 0인덱스부터 15인덱스까지 가져와 grid 형식으로 출력합니다. 
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(16))) # range(16)은 0부터 15까지 반복해서 classes 리스트를 출력합니다. 

# 4. CNN 모델 정의 함수 (명명된 인자 사용)
def define_model() -> nn.Module:
    """
    CNN 모델을 정의합니다.
    
    Returns:
        nn.Module: 정의된 CNN 모델
    """
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # 첫 번째 Convolutional Layer - (64,3,32,32) -> (64,32,32,32)
            # 역할: 입력 이미지에서 기본적인 특징(예: 에지, 선 등)을 추출
            self.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            )
            # 역할: 각 배치마다 정규화를 수행하여 학습을 안정화하고, 학습 속도를 향상
            self.bn1 = nn.BatchNorm2d(num_features=32) # PyTorch에서 2D 배치 정규화(Batch Normalization)를 수행하는 레이어를 정의
            
            # 두 번째 Convolutional Layer
            # 역할: 첫 번째 Convolutional Layer에서 추출한 기본적인 특징을 더 복잡하고 추상적인 특징으로 변환
            self.conv2 = nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            )
            # 역할: 두 번째 Convolutional Layer의 출력을 정규화하여 학습을 더욱 안정화
            self.bn2 = nn.BatchNorm2d(num_features=32)
            
            # Max Pooling
            # 역할: 공간적 크기를 절반으로 줄이고, 가장 중요한 특징만을 추출하여 계산량 감소
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            # 역할: 과적합(overfitting)을 방지하기 위해 일부 뉴런을 무작위로 비활성화
            self.dropout = nn.Dropout(p=0.25)
            
            # 세 번째 Convolutional Layer
            # 역할: 첫 번째 블록에서 추출한 특징을 더 높은 수준의 추상적인 특징으로 변환
            self.conv3 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            )
            # 역할: 세 번째 Convolutional Layer의 출력을 정규화하여 학습을 안정화
            self.bn3 = nn.BatchNorm2d(num_features=64)
            
            # 네 번째 Convolutional Layer
            # 역할: 더 높은 수준의 특징을 더욱 복잡하도록 변환
            self.conv4 = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            )
            # 역할: 네 번째 Convolutional Layer의 출력을 정규화하여 학습을 더욱 안정화
            self.bn4 = nn.BatchNorm2d(num_features=64)
            # 역할: 두 번째 블록에서도 과적합을 방지하기 위해 일부 뉴런을 무작위로 비활성화
            self.dropout2 = nn.Dropout(p=0.25)
            
            # Fully Connected Layers : 완전 연결층은 추출된 특징을 기반으로 최종 클래스에 대한 예측을 수행
            # 역할: 추출된 특징을 고차원 공간으로 매핑하여 복잡한 패턴을 학습
            self.fc1 = nn.Linear(
                in_features=64 * 8 * 8,
                out_features=512
            )
            # 역할: 첫 번째 Fully Connected Layer의 출력을 정규화하여 학습을 안정화
            self.bn5 = nn.BatchNorm1d(num_features=512)
            # 역할: 과적합을 더욱 효과적으로 방지하기 위해 높은 확률(50%)로 일부 뉴런을 무작위로 비활성화
            self.dropout3 = nn.Dropout(p=0.5)
            # 역할: 최종 클래스에 대한 예측을 수행
            self.fc2 = nn.Linear(
                in_features=512,
                out_features=10
            )
        
        def forward(self, x):
            # 첫 번째 블록
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = self.dropout(x)
            
            # 두 번째 블록
            x = torch.relu(self.bn3(self.conv3(x)))
            x = torch.relu(self.bn4(self.conv4(x)))
            x = self.pool(x)
            x = self.dropout2(x)
            
            # 완전 연결층
            x = x.view(-1, 64 * 8 * 8)
            x = torch.relu(self.bn5(self.fc1(x)))
            x = self.dropout3(x)
            x = self.fc2(x)
            return x
    
    net = Net()
    print(net)
    return net

# 5. 손실 함수 및 옵티마이저 정의 함수
def get_criterion_optimizer(model: nn.Module, learning_rate: float = 0.001) -> tuple:
    """
    손실 함수와 옵티마이저를 정의합니다.
    
    Args:
        model (nn.Module): 학습할 모델
        learning_rate (float, optional): 학습률. Defaults to 0.001.
    
    Returns:
        tuple: (손실 함수, 옵티마이저)
    """
    criterion = nn.CrossEntropyLoss() # 손실 함수 객체 생성
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate) # 옵티마이저 객체 생성
    return criterion, optimizer

# 6. 모델 학습 함수
def train_model(model: nn.Module, 
                trainloader: DataLoader, 
                valloader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                device: torch.device, 
                num_epochs: int = 50) -> dict:
    """
    모델을 학습시킵니다.
    
    Args:
        model (nn.Module): 학습할 모델
        trainloader (DataLoader): 학습 데이터 로더
        valloader (DataLoader): 검증 데이터 로더
        criterion (nn.Module): 손실 함수
        optimizer (optim.Optimizer): 옵티마이저
        device (torch.device): 사용 장치
        num_epochs (int, optional): 학습할 에포크 수. Defaults to 50.
    
    Returns:
        dict: 학습 및 검증 손실과 정확도 기록
    """
    history = {
        'train_losses': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': []
    }
    
    for epoch in range(num_epochs):  # 에포크 수
        # 학습 단계
        model.train() # 훈련 모드 
        # --- 평가 지표 ---
        running_loss = 0.0
        correct = 0
        total = 0
        # ---
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  # 기울기 초기화
            
            outputs = model(inputs)  # 순전파
            loss = criterion(outputs, labels)  # 손실 계산
            loss.backward()  # 역전파
            optimizer.step()  # 가중치 업데이트
            
            running_loss += loss.item() * inputs.size(0) # loss 
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = correct / total
        history['train_losses'].append(epoch_loss)
        history['train_accuracies'].append(epoch_acc)
        
        # 검증 단계
        model.eval()
        val_loss = 0.0
        correct_test = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(valloader.dataset)
        epoch_val_acc = correct_test / total_val
        history['val_losses'].append(epoch_val_loss)
        history['val_accuracies'].append(epoch_val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} '
              f'| Test Loss: {epoch_val_loss:.4f} | Test Acc: {epoch_val_acc:.4f}')
    
    print('train complete')
    return history

# 7. 모델 평가 함수
def evaluate_model(model: nn.Module, 
                   testloader: DataLoader, 
                   device: torch.device) -> float:
    """
    테스트 데이터셋을 사용하여 모델을 평가합니다.
    
    Args:
        model (nn.Module): 평가할 모델
        testloader (DataLoader): 테스트 데이터 로더
        device (torch.device): 사용 장치
    
    Returns:
        float: 테스트 정확도
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = correct / total
    print(f'\nfinal test accuracy: {test_accuracy:.4f}')
    return test_accuracy

# 8. 학습 과정 시각화 함수
def plot_history(history: dict, num_epochs: int):
    """
    학습 과정에서의 손실과 정확도를 시각화합니다.
    
    Args:
        history (dict): 학습 및 검증 손실과 정확도 기록
        num_epochs (int): 에포크 수
    """
    epochs = range(1, num_epochs + 1)
    
    plt.figure(figsize=(12,5))
    
    # 정확도 변화
    plt.subplot(1,2,1) # (행 수, 열 수, 서브플롯 위치)
    plt.plot(epochs, history['train_accuracies'], label='train accuracy')
    plt.plot(epochs, history['val_accuracies'], label='val accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy change')
    plt.legend()
    
    # 손실 변화
    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_losses'], label='train loss')
    plt.plot(epochs, history['val_losses'], label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss change')
    plt.legend()
    
    plt.show()

# 9. 메인 함수
def main():
    # 장치 설정
    device = get_device()
    
    # 데이터 로드
    trainloader, valloader, testloader, classes = load_data()
    
    # 데이터 시각화
    visualize_data(trainloader, classes)
    
    # 모델 정의
    model = define_model().to(device)
    
    # 손실 함수와 옵티마이저 정의
    criterion, optimizer = get_criterion_optimizer(model)
    
    # 모델 학습
    num_epochs = 10
    history = train_model(model, trainloader, valloader, criterion, optimizer, device, num_epochs)
    
    # 모델 평가
    evaluate_model(model, testloader, device)
    
    # 학습 과정 시각화
    plot_history(history, num_epochs)

if __name__ == '__main__':
    main()