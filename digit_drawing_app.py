# MNIST 손글씨 숫자 인식 GUI 애플리케이션
import tkinter as tk
from tkinter import Canvas, Button, Label, Frame, Scale
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import cv2
import tensorflow as tf
from scipy import ndimage

class DigitDrawingApp:
    
    def __init__(self, window):
        self.window = window
        self.window.title("MNIST 손글씨 숫자 인식기")
        self.window.configure(bg='#f0f0f0')
        
        # 모델 로드
        self.model = self.load_model()
        
        # 그리기 상태 변수
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
        self.drawing_history = []
        
        # 캔버스 크기 (MNIST와 동일한 비율 유지)
        self.canvas_width = 280
        self.canvas_height = 280
        
        # PIL 이미지 (실제 그리기용)
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        # GUI 구성
        self.setup_gui()
        
        # 실시간 예측 제거됨
        
    def load_model(self):
        """저장된 모델 로드"""
        try:
            model = tf.keras.models.load_model('mnist_model.h5')
            print("모델이 성공적으로 로드되었습니다.")
            return model
        except:
            print("경고: 모델 파일을 찾을 수 없습니다. 먼저 mnist_classifier_model.py를 실행하여 모델을 학습시켜주세요.")
            return None
        
    def setup_gui(self):
        """GUI 컴포넌트 설정"""
        # 제목 라벨
        title_label = Label(self.window, text="손글씨 숫자 인식기", 
                          font=("Arial", 20, "bold"), bg='#f0f0f0', fg='black')
        title_label.pack(pady=10)
        
        # 설명 라벨
        instruction_label = Label(self.window, text="아래 캔버스에 0-9 사이의 숫자를 그려주세요", 
                          font=("Arial", 10), bg='#f0f0f0', fg='gray')
        instruction_label.pack(pady=5)
        
        # 캔버스 프레임
        canvas_frame = Frame(self.window, bg='#d0d0d0', bd=3, relief=tk.RAISED)
        canvas_frame.pack(pady=10)
        
        # 그리기 캔버스
        self.canvas = Canvas(canvas_frame, width=self.canvas_width, 
                           height=self.canvas_height, bg='white', 
                           cursor="pencil", highlightthickness=0)
        self.canvas.pack(padx=5, pady=5)
        
        # 캔버스 이벤트 바인딩
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        # 결과 표시 프레임
        result_frame = Frame(self.window, bg='#f0f0f0')
        result_frame.pack(pady=10)
        
        Label(result_frame, text="예측 결과:", font=("Arial", 16), bg='#f0f0f0', fg='black').pack(side=tk.LEFT)
        self.result_label = Label(result_frame, text="?", font=("Arial", 48, "bold"), 
                                fg="blue", width=3, bg='#f0f0f0')
        self.result_label.pack(side=tk.LEFT, padx=20)
        
        # 신뢰도 표시 라벨
        self.confidence_label = Label(self.window, text="", font=("Arial", 12), 
                                    bg='#f0f0f0', fg='black')
        self.confidence_label.pack(pady=5)
        
        # 버튼 프레임
        button_frame = Frame(self.window, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        # 지우기 버튼
        clear_button = Button(button_frame, text="지우기", command=self.clear_canvas,
                         bg="#ff6b6b", fg="black", relief=tk.RAISED, bd=3,
                         activebackground="#ff5252", activeforeground="black")
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # 예측 버튼
        self.predict_button = Button(button_frame, text="예측하기", command=self.predict_digit,
                           bg="#4CAF50", fg="black", relief=tk.RAISED, bd=3,
                           activebackground="#45a049", activeforeground="black")
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        # 브러시 크기 조절
        brush_frame = Frame(self.window, bg='#f0f0f0')
        brush_frame.pack(pady=5)
        
        Label(brush_frame, text="브러시 크기:", bg='#f0f0f0', fg='black').pack(side=tk.LEFT)
        self.brush_size = Scale(brush_frame, from_=5, to=20, orient=tk.HORIZONTAL,
                              bg='#f0f0f0', troughcolor='#d0d0d0', fg='black')
        self.brush_size.set(12)
        self.brush_size.pack(side=tk.LEFT)
        
    def clear_canvas(self):
        """캔버스와 이미지 초기화"""
        self.canvas.delete("all")
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="?", fg="blue")
        self.confidence_label.config(text="")
        self.drawing_history = []
        
    def start_drawing(self, event):
        """그리기 시작"""
        self.is_drawing = True
        self.last_x = event.x
        self.last_y = event.y
        
    def draw_line(self, event):
        """선 그리기 (개선된 부드러운 선)"""
        if self.is_drawing and self.last_x is not None and self.last_y is not None:
            brush_size = self.brush_size.get()
            
            # 부드러운 선을 위한 중간점 계산
            distance = np.sqrt((event.x - self.last_x)**2 + (event.y - self.last_y)**2)
            if distance > 0:
                steps = max(int(distance), 1)
                for i in range(steps):
                    t = i / steps
                    x = int(self.last_x + t * (event.x - self.last_x))
                    y = int(self.last_y + t * (event.y - self.last_y))
                    
                    # 캔버스에 원 그리기
                    self.canvas.create_oval(
                        x - brush_size//2, y - brush_size//2,
                        x + brush_size//2, y + brush_size//2,
                        fill='black', outline='black'
                    )
                    
                    # PIL 이미지에도 그리기 (안티앨리어싱 효과)
                    self.draw.ellipse(
                        [x - brush_size//2, y - brush_size//2,
                         x + brush_size//2, y + brush_size//2],
                        fill='black'
                    )
            
            self.last_x = event.x
            self.last_y = event.y
                    
    def stop_drawing(self, event):
        """그리기 종료"""
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
        
            
    def preprocess_image(self):
        """이미지 전처리 (MNIST 형식으로 변환)"""
        # 그레이스케일 이미지를 numpy 배열로 변환
        img_array = np.array(self.image)
        
        # 이미지가 비어있는지 확인
        if np.all(img_array == 255):
            return None
            
        # 숫자가 있는 부분만 추출 (바운딩 박스)
        coords = np.column_stack(np.where(img_array < 255))
        if len(coords) == 0:
            return None
            
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        
        # 여백 추가
        padding = 20
        y0 = max(0, y0 - padding)
        x0 = max(0, x0 - padding)
        y1 = min(img_array.shape[0], y1 + padding)
        x1 = min(img_array.shape[1], x1 + padding)
        
        # 숫자 부분만 추출
        cropped = img_array[y0:y1, x0:x1]
        
        # 정사각형으로 만들기
        height, width = cropped.shape
        if height > width:
            pad_width = (height - width) // 2
            cropped = np.pad(cropped, ((0, 0), (pad_width, pad_width)), 
                           mode='constant', constant_values=255)
        else:
            pad_height = (width - height) // 2
            cropped = np.pad(cropped, ((pad_height, pad_height), (0, 0)), 
                           mode='constant', constant_values=255)
        
        # 28x28로 리사이즈
        resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)
        
        # 색상 반전 (흰 배경, 검은 글씨 -> 검은 배경, 흰 글씨)
        inverted = 255 - resized
        
        # 정규화 (0-1 범위)
        normalized = inverted.astype('float32') / 255.0
        
        # 가우시안 필터로 부드럽게
        smoothed = ndimage.gaussian_filter(normalized, sigma=0.5)
        
        # CNN 입력 형태로 변환
        return smoothed.reshape(1, 28, 28, 1)
    
    def predict_digit(self):
        """숫자 예측"""
        if self.model is None:
            self.result_label.config(text="!", fg="red")
            self.confidence_label.config(text="모델이 로드되지 않았습니다")
            return
            
        # 이미지 전처리
        processed_image = self.preprocess_image()
        
        if processed_image is None:
            return
        
        # 예측
        prediction = self.model.predict(processed_image, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        # 결과 표시
        if confidence > 90:
            color = "green"
        elif confidence > 70:
            color = "orange"
        else:
            color = "red"
        self.result_label.config(text=str(predicted_digit), fg=color)
        self.confidence_label.config(text=f"신뢰도: {confidence:.1f}%")
        
def main():
    """메인 실행 함수"""
    root = tk.Tk()
    app = DigitDrawingApp(root)
    root.mainloop()
    
if __name__ == "__main__":
    main()