# 개선된 MNIST 손글씨 숫자 인식을 위한 CNN 분류기
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class ImprovedMNISTClassifier:
    
    def __init__(self):
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self):
        """MNIST 데이터를 로드하고 전처리"""
        # MNIST 데이터셋 로드
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # 픽셀 값을 0-1 범위로 정규화
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # CNN 입력을 위해 채널 차원 추가
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # 라벨을 원-핫 인코딩으로 변환
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        
        return (x_train, y_train), (x_test, y_test)
    
    def load_user_data(self):
        """사용자가 수집한 데이터를 로드"""
        try:
            user_images = np.load('user_data_images.npy')
            user_labels = np.load('user_data_labels.npy')
            
            # 사용자 데이터 형태 확인 및 조정
            if len(user_images.shape) == 3:  # (N, 28, 28)
                user_images = user_images.reshape(-1, 28, 28, 1)  # (N, 28, 28, 1)
            
            # 라벨을 원-핫 인코딩으로 변환
            user_labels_onehot = to_categorical(user_labels, 10)
            
            print(f"사용자 데이터 로드 완료: {len(user_images)}개")
            return user_images, user_labels_onehot
            
        except FileNotFoundError:
            print("사용자 데이터 파일이 없습니다. MNIST 데이터만 사용합니다.")
            return None, None
        except Exception as e:
            print(f"사용자 데이터 로드 중 오류: {e}")
            return None, None
    
    def combine_data(self, mnist_data, user_data):
        """MNIST 데이터와 사용자 데이터를 결합"""
        (x_train, y_train), (x_test, y_test) = mnist_data
        user_images, user_labels = user_data
        
        if user_images is not None and user_labels is not None:
            # 사용자 데이터를 훈련 데이터에 추가
            x_train_combined = np.concatenate([x_train, user_images], axis=0)
            y_train_combined = np.concatenate([y_train, user_labels], axis=0)
            
            print(f"데이터 결합 완료:")
            print(f"  기존 MNIST 훈련 데이터: {len(x_train)}개")
            print(f"  사용자 데이터: {len(user_images)}개")
            print(f"  결합된 훈련 데이터: {len(x_train_combined)}개")
            
            return (x_train_combined, y_train_combined), (x_test, y_test)
        else:
            return mnist_data
    
    def build_model(self):
        """CNN 모델 구성"""
        self.model = models.Sequential([
            # 첫 번째 합성곱 블록
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 두 번째 합성곱 블록
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 세 번째 합성곱 블록
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),  # Flatten 대신 사용
            
            # 완전 연결 층
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # 출력 층
            layers.Dense(10, activation='softmax')
        ])
        
        # 모델 컴파일
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def create_data_augmentation(self):
        """데이터 증강 생성기 생성"""
        datagen = ImageDataGenerator(
            rotation_range=10,        # 회전
            width_shift_range=0.1,    # 수평 이동
            height_shift_range=0.1,   # 수직 이동
            shear_range=0.1,          # 전단 변환
            zoom_range=0.1,           # 확대/축소
            fill_mode='nearest'       # 빈 픽셀 채우기
        )
        return datagen
    
    def train_with_augmentation(self, x_train, y_train, x_val, y_val, epochs=30, batch_size=128):
        """데이터 증강을 사용한 모델 훈련"""
        # 데이터 증강 생성기
        datagen = self.create_data_augmentation()
        datagen.fit(x_train)
        
        # 콜백 설정
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )
        
        # 모델 훈련
        self.history = self.model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(x_train) // batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, x_test, y_test):
        """테스트 데이터로 모델 성능 평가"""
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        return test_loss, test_accuracy
    
    def predict(self, x):
        """이미지 배치에 대한 예측 수행"""
        predictions = self.model.predict(x)
        return np.argmax(predictions, axis=1)
    
    def predict_single(self, image):
        """단일 이미지에 대한 예측 수행"""
        if len(image.shape) == 2:
            image = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 3:
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        
        prediction = self.model.predict(image)
        return np.argmax(prediction, axis=1)[0]
    
    def plot_training_history(self):
        """훈련 과정의 정확도와 손실 그래프 출력"""
        if self.history is None:
            print("모델이 아직 훈련되지 않았습니다.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 정확도 그래프
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # 손실 그래프
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """훈련된 모델을 파일로 저장"""
        if self.model is None:
            print("저장할 모델이 없습니다.")
            return
        self.model.save(filepath)
        print(f"모델이 {filepath}에 저장되었습니다.")
    
    def load_model(self, filepath):
        """저장된 모델을 파일에서 로드"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"모델이 {filepath}에서 로드되었습니다.")
    
    def test_with_custom_images(self, x_test, y_test, num_images=10):
        """커스텀 이미지 테스트"""
        indices = np.random.choice(len(x_test), num_images, replace=False)
        test_images = x_test[indices]
        test_labels = y_test[indices]
        predictions = self.model.predict(test_images)
        predicted_classes = np.argmax(predictions, axis=1)
        actual_classes = np.argmax(test_labels, axis=1)
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i in range(num_images):
            axes[i].imshow(test_images[i].reshape(28, 28), cmap='gray')
            axes[i].axis('off')
            
            confidence = np.max(predictions[i]) * 100
            
            if predicted_classes[i] == actual_classes[i]:
                color = 'green' if confidence > 90 else 'orange'
            else:
                color = 'red'
            
            axes[i].set_title(
                f'예측: {predicted_classes[i]} ({confidence:.1f}%)\n실제: {actual_classes[i]}',
                color=color,
                fontsize=10
            )
        
        plt.tight_layout()
        plt.show()

def main():
    """메인 실행 함수"""
    print("개선된 MNIST 필기 숫자 인식 프로그램을 시작합니다...")
    
    # 분류기 인스턴스 생성
    classifier = ImprovedMNISTClassifier()
    
    # 1. MNIST 데이터 로드 및 전처리
    print("MNIST 데이터를 로드하고 전처리합니다...")
    mnist_data = classifier.load_and_preprocess_data()
    
    # 2. 사용자 데이터 로드
    print("\n사용자 데이터를 확인합니다...")
    user_data = classifier.load_user_data()
    
    # 3. 데이터 결합
    print("\n데이터를 결합합니다...")
    combined_data = classifier.combine_data(mnist_data, user_data)
    (x_train, y_train), (x_test, y_test) = combined_data
    
    # 훈련 데이터의 일부를 검증 데이터로 분리
    val_split = int(0.1 * len(x_train))
    x_val = x_train[-val_split:]
    y_val = y_train[-val_split:]
    x_train = x_train[:-val_split]
    y_train = y_train[:-val_split]
    
    print(f"\n최종 데이터 형태:")
    print(f"훈련 데이터: {x_train.shape}")
    print(f"검증 데이터: {x_val.shape}")
    print(f"테스트 데이터: {x_test.shape}")
    
    # 4. 모델 구성
    print("\n개선된 모델을 구성합니다...")
    model = classifier.build_model()
    model.summary()
    
    # 5. 모델 훈련 (데이터 증강 포함)
    print("\n데이터 증강을 사용하여 모델을 훈련합니다...")
    history = classifier.train_with_augmentation(
        x_train, y_train, x_val, y_val, 
        epochs=20,  # Early stopping으로 자동 조절
        batch_size=128
    )
    
    # 6. 모델 성능 평가
    print("\n모델을 평가합니다...")
    test_loss, test_accuracy = classifier.evaluate(x_test, y_test)
    print(f"테스트 정확도: {test_accuracy:.4f}")
    print(f"테스트 손실: {test_loss:.4f}")
    
    # 7. 모델 저장
    print("\n모델을 저장합니다...")
    classifier.save_model('mnist_model.h5')
    
    # 8. 훈련 과정 시각화
    classifier.plot_training_history()
    
    # 9. 테스트 샘플 예측
    print("\n테스트 샘플로 예측을 수행합니다...")
    classifier.test_with_custom_images(x_test, y_test)
    
if __name__ == "__main__":
    main()