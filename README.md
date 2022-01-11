# aiffel_E01
Exploration_1, rock-scissor-paper using Tensorflow CNN    
AIFFEL교육과정 중 첫번째 Exploration으로 가위, 바위, 보 이미지 학습 후 검증

## 개요
본 시스템은 크게 3가지 단계로 이루어져 있습니다.
1. Data pre-processing
/data에 있는 이미지를 가져옵니다.
라벨(가위 : 0, 바위 : 1, 보 : 2)을 붙입니다.
이미지 픽셀의 범위를 (0-255) -> (0-1)로 바꿉니다.
2. Deep learning
데이터를 CNN을 통해 학습합니다.
3. Predict
학습한 모델을 통해서 Test이미지가 어떤 손동작인지 예측합니다.

## Installation
파이썬 개발 환경으로 최신 버전의 Anaconda를 설치하세요. (Python3 버전용)
* tensorflow (2 이상)
* numpy
* matplotlib
* PIL

```
$ pip install -r requirements.txt
```

------------
## Directory
필수 디렉토리는 다음과 같습니다
```
.
├── imgs/
├── data/
│   ├── train/
│   │   ├── rock/
│   │   ├── scissor/
│   │   └── paper/
│   └── test/
│       ├── rock/
│       ├── scissor/
│       └── paper/
│
├── main.py
└── main.py
```
시작하기 앞서 각 폴더안에는 한글파일, 폴더와 상관없이 모든 이미지(.png, .PNG, .jpg, .JPG)를 인식할 수 있습니다.

------------
### model

![model](./img/model.png)

### loss

![loss](./img/loss.png)

### result

![result](./img/result.png)

------------
## hyper parameter 수정 내용
default : 0.3~0.4(accuarcy)
(28, 28, 3), trainset:100, epochs=5

mk1: 0.5~0.6
(224, 224, 3), trainset:5627, epochs=5

mk2: 0.6-0.7, 0.5-0.6, 0.7-0.8
(28, 28, 3), trainset:5627, epochs=5

mk3: 0.5~0.6
(28, 28, 1), trainset:5627, epochs=5

mk4: 0.5~0.6
(28, 28, 3), trainset:5627, epochs=10

mk5: 0.6~0.7
(28, 28, 3), trainset:5627, epochs=10, Dense layer(512) 1개 삭제

mk6: 0.6~0.7
(56, 56, 3), trainset:5627, epochs=6

------------
## 차별점, 문제점
1. LMS에서는 한 라벨당 100장의 이미지으로 제한되었지만 이미지뿐만 아니라 폴더, 이미지수에 관계없이 인식할 수 있습니다.
2. 원한다면 GRAY에서 작업할 수 있습니다.
3. system에 관계없이 인식할 수 있게 했습니다.
4. train image를 shuffle하는 기능을 추가했습니다.
5. test acc가 70%를 넘겼습니다.
6. 하지만 test result를 보면 loss값이 너무 큰 문제가 있습니다.
7. train acc와 test acc차이가 크므로 overfitting이 의심됩니다.
    - 6, 7를 해결하기 위해 데이터 전처리를 추가해야할 것 같습니다. 
    - 예를 들면 피부색만 인식, 경계선 검출등을 이용

