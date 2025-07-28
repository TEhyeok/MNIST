# MNIST 숫자 인식 모델
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
        """데이터 불러와서 정리하기"""
        # 기본 데이터 불러오기
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # 0~1 범위로 맞추기
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # 모델용 형태로 바꾸기
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # 라벨도 모델용으로
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        
        return (x_train, y_train), (x_test, y_test)
    
    def load_user_data(self):
        """내가 그린 데이터 불러오기"""
        try:
            user_images = np.load('user_data_images.npy')
            user_labels = np.load('user_data_labels.npy')
            
            # 데이터 형태 맞추기
            if len(user_images.shape) == 3:  # (N, 28, 28)
                user_images = user_images.reshape(-1, 28, 28, 1)  # (N, 28, 28, 1)
            
            # 라벨도 모델용으로
            user_labels_onehot = to_categorical(user_labels, 10)
            
            print(f"사용자 데이터 로드 완료: {len(user_images)}개")
            return user_images, user_labels_onehot
            
        except FileNotFoundError:
            print("내 데이터가 없네. 기본 데이터만 쓸게.")
            return None, None
        except Exception as e:
            print(f"사용자 데이터 로드 중 오류: {e}")
            return None, None
    
    def combine_data(self, mnist_data, user_data):
        """기본 데이터랑 내 데이터 합치기"""
        (x_train, y_train), (x_test, y_test) = mnist_data
        user_images, user_labels = user_data
        
        if user_images is not None and user_labels is not None:
            # 내 데이터도 같이 넣기
            x_train_combined = np.concatenate([x_train, user_images], axis=0)
            y_train_combined = np.concatenate([y_train, user_labels], axis=0)
            
            print(f"데이터 합치기 완료:")
            print(f"  기본 데이터: {len(x_train)}개")
            print(f"  내 데이터: {len(user_images)}개")
            print(f"  총 데이터: {len(x_train_combined)}개")
            
            return (x_train_combined, y_train_combined), (x_test, y_test)
        else:
            return mnist_data
    
    def build_model(self):
        """모델 만들기"""
        self.model = models.Sequential([
            # 첫번째 층
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 두번째 층
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # 세번째 층
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # 마지막 연결부분
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # 최종 결과
            layers.Dense(10, activation='softmax')
        ])
        
        # 모델 세팅
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def create_data_augmentation(self):
        """데이터 늘리기"""
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        return datagen
    
    def train_with_augmentation(self, x_train, y_train, x_val, y_val, epochs=30, batch_size=128):
        """모델 학습하기"""
        # 데이터 늘리기
        datagen = self.create_data_augmentation()
        datagen.fit(x_train)
        
        # 학습 옵션들
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
        
        # 실제 학습
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
        """모델 성능 체크"""
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        return test_loss, test_accuracy
    
    def predict(self, x):
        """여러개 한번에 예측"""
        predictions = self.model.predict(x)
        return np.argmax(predictions, axis=1)
    
    def predict_single(self, image):
        """하나만 예측"""
        if len(image.shape) == 2:
            image = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 3:
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        
        prediction = self.model.predict(image)
        return np.argmax(prediction, axis=1)[0]
    
    def plot_training_history(self):
        """학습 과정 그래프"""
        if self.history is None:
            print("아직 학습 안했어.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 정확도
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # 오류율
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
        """모델 저장"""
        if self.model is None:
            print("저장할게 없어.")
            return
        self.model.save(filepath)
        print(f"모델이 {filepath}에 저장되었습니다.")
    
    def load_model(self, filepath):
        """모델 불러오기"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"모델이 {filepath}에서 로드되었습니다.")
    
    def test_with_custom_images(self, x_test, y_test, num_images=10):
        """테스트 이미지들로 확인"""
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
    """실행"""
    print("MNIST 숫자 인식 프로그램 시작...")
    
    # 분류기 만들기
    classifier = ImprovedMNISTClassifier()
    
    # 1. 기본 데이터 불러오기
    print("기본 데이터 불러오는중...")
    mnist_data = classifier.load_and_preprocess_data()
    
    # 2. 내 데이터 확인
    print("\n내 데이터 있나 확인중...")
    user_data = classifier.load_user_data()
    
    # 3. 데이터 합치기
    print("\n데이터 합치는중...")
    combined_data = classifier.combine_data(mnist_data, user_data)
    (x_train, y_train), (x_test, y_test) = combined_data
    
    # 검증용 데이터 따로 빼기
    val_split = int(0.1 * len(x_train))
    x_val = x_train[-val_split:]
    y_val = y_train[-val_split:]
    x_train = x_train[:-val_split]
    y_train = y_train[:-val_split]
    
    print(f"\n최종 데이터:")
    print(f"훈련 데이터: {x_train.shape}")
    print(f"검증 데이터: {x_val.shape}")
    print(f"테스트 데이터: {x_test.shape}")
    
    # 4. 모델 만들기
    print("\n모델 만드는중...")
    model = classifier.build_model()
    model.summary()
    
    # 5. 모델 학습
    print("\n학습 시작...")
    history = classifier.train_with_augmentation(
        x_train, y_train, x_val, y_val, 
        epochs=20,
        batch_size=128
    )
    
    # 6. 성능 체크
    print("\n성능 확인중...")
    test_loss, test_accuracy = classifier.evaluate(x_test, y_test)
    print(f"테스트 정확도: {test_accuracy:.4f}")
    print(f"테스트 손실: {test_loss:.4f}")
    
    # 7. 모델 저장
    print("\n모델 저장중...")
    classifier.save_model('mnist_model.h5')
    
    # 8. 그래프 보기
    classifier.plot_training_history()
    
    # 9. 테스트
    print("\n테스트 해보기...")
    classifier.test_with_custom_images(x_test, y_test)
    
if __name__ == "__main__":
    main()