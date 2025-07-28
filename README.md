# MNIST 숫자 인식

손글씨 숫자를 인식하는 프로그램입니다.

## 파일 설명

- `mnist_classifier_model.py` - 모델 학습용
- `digit_drawing_app.py` - 숫자 인식 GUI
- `data_collection_app.py` - 데이터 수집용 GUI
- `requirements.txt` - 필요한 패키지들

## 사용법

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 모델 학습
```bash
python mnist_classifier_model.py
```

### 3. GUI 실행
```bash
python digit_drawing_app.py
```

### 4. 데이터 수집 (선택사항)
```bash
python data_collection_app.py
```
숫자를 그리고 정답을 선택해서 데이터를 모을 수 있습니다.
모은 데이터는 다음 학습에 자동으로 포함됩니다.

## 특징

- CNN 모델 사용
- 실시간 GUI 인식
- 사용자 데이터 추가 학습 가능
- 브러시 크기 조절 가능

## 요구사항

- Python 3.7+
- TensorFlow 2.10+
- NumPy
- Matplotlib
- Tkinter (기본 포함)
- PIL
- OpenCV
- SciPy