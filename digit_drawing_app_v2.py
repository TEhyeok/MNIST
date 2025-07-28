# 개선된 손글씨 숫자 그리기 및 실시간 인식 애플리케이션
import tkinter as tk
from tkinter import Canvas, Button, Label, Frame, Scale
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import cv2
from mnist_classifier import MNISTClassifier
import tensorflow as tf
from scipy import ndimage
import threading
import time

class ImprovedDigitDrawingApp:
    """개선된 손글씨 숫자 인식 GUI 애플리케이션"""
    
    def __init__(self, master):
        """애플리케이션 초기화"""
        self.master = master
        self.master.title("개선된 MNIST 손글씨 숫자 인식")
        self.master.geometry("600x800")
        self.master.resizable(False, False)
        self.master.configure(bg='#f0f0f0')
        
        # 캔버스 설정
        self.canvas_width = 400
        self.canvas_height = 400
        self.default_brush_width = 15  # 브러시 크기 줄임
        self.brush_width = self.default_brush_width
        
        # 그리기 상태
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.preview_window = None
        self.stroke_points = []  # 획 포인트 저장
        
        # 실시간 예측 설정
        self.realtime_prediction = True
        self.last_prediction_time = 0
        self.prediction_delay = 0.5  # 0.5초마다 예측
        
        # PIL 이미지 (캔버스 내용 저장용)
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        # MNIST 분류기 로드
        self.classifier = MNISTClassifier()
        self.load_model()
        
        # UI 구성
        self.setup_ui()
        
        # 실시간 예측 스레드 시작
        if self.realtime_prediction and self.model_loaded:
            self.start_realtime_prediction()
        
    def load_model(self):
        """저장된 MNIST 모델 로드"""
        try:
            self.classifier.load_model('mnist_model.h5')
            print("모델이 성공적으로 로드되었습니다.")
            self.model_loaded = True
        except:
            print("모델을 찾을 수 없습니다. 먼저 mnist_classifier.py를 실행하여 모델을 훈련시켜주세요.")
            self.model_loaded = False
    
    def setup_ui(self):
        """UI 구성 요소 설정"""
        # 제목 라벨
        title_label = Label(self.master, text="숫자를 그려주세요 (0-9)", 
                          font=("Arial", 20, "bold"), bg='#f0f0f0', fg='black')
        title_label.pack(pady=10)
        
        # 안내 라벨
        info_label = Label(self.master, text="팁: MNIST 스타일로 그리세요 - 중앙에 크게, 약간 기울여서", 
                          font=("Arial", 10), bg='#f0f0f0', fg='gray')
        info_label.pack()
        
        # 캔버스
        self.canvas = Canvas(self.master, width=self.canvas_width, 
                           height=self.canvas_height, bg='white', 
                           highlightthickness=2, highlightbackground="black")
        self.canvas.pack(pady=10)
        
        # 마우스 이벤트 바인딩
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_motion)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)
        
        # 브러시 크기 조절 슬라이더
        brush_frame = Frame(self.master, bg='#f0f0f0')
        brush_frame.pack()
        Label(brush_frame, text="브러시 크기:", font=("Arial", 10), bg='#f0f0f0').pack(side=tk.LEFT)
        self.brush_slider = Scale(brush_frame, from_=5, to=25, orient=tk.HORIZONTAL, 
                                 command=self.update_brush_size, bg='#f0f0f0')
        self.brush_slider.set(self.default_brush_width)
        self.brush_slider.pack(side=tk.LEFT)
        
        # 예측 결과 표시 프레임
        result_frame = Frame(self.master, bg='#f0f0f0')
        result_frame.pack(pady=10)
        
        Label(result_frame, text="예측 결과:", font=("Arial", 16), bg='#f0f0f0', fg='black').pack(side=tk.LEFT)
        self.result_label = Label(result_frame, text="?", font=("Arial", 36, "bold"), 
                                fg="blue", width=3, bg='#f0f0f0')
        self.result_label.pack(side=tk.LEFT, padx=10)
        
        # 신뢰도 표시
        self.confidence_label = Label(self.master, text="신뢰도: 0%", 
                                    font=("Arial", 14), bg='#f0f0f0')
        self.confidence_label.pack()
        
        # 확률 표시 라벨
        self.prob_label = Label(self.master, text="", font=("Arial", 10), 
                               bg='#f0f0f0', justify=tk.LEFT, height=5)
        self.prob_label.pack()
        
        # 버튼 프레임
        button_frame = Frame(self.master, bg='#f0f0f0')
        button_frame.pack(pady=20)
        
        # 지우기 버튼
        clear_btn = Button(button_frame, text="지우기", command=self.clear_canvas,
                         font=("Arial", 14, "bold"), width=12, height=2,
                         bg="#ff6b6b", fg="black", relief=tk.RAISED, bd=3)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # 예측 버튼
        predict_btn = Button(button_frame, text="예측하기", command=self.predict_digit,
                           font=("Arial", 14, "bold"), width=12, height=2, 
                           bg="#4CAF50", fg="black", relief=tk.RAISED, bd=3)
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        # 실시간 예측 토글
        self.realtime_var = tk.BooleanVar(value=True)
        realtime_check = tk.Checkbutton(self.master, text="실시간 예측", 
                                       variable=self.realtime_var,
                                       command=self.toggle_realtime,
                                       font=("Arial", 10), bg='#f0f0f0')
        realtime_check.pack()
        
    def update_brush_size(self, value):
        """브러시 크기 업데이트"""
        self.brush_width = int(value)
        
    def toggle_realtime(self):
        """실시간 예측 토글"""
        self.realtime_prediction = self.realtime_var.get()
        if self.realtime_prediction and self.model_loaded:
            self.start_realtime_prediction()
            
    def start_realtime_prediction(self):
        """실시간 예측 스레드 시작"""
        def predict_loop():
            while self.realtime_prediction:
                current_time = time.time()
                if current_time - self.last_prediction_time > self.prediction_delay:
                    if self.has_content():
                        self.predict_digit(silent=True)
                    self.last_prediction_time = current_time
                time.sleep(0.1)
        
        thread = threading.Thread(target=predict_loop, daemon=True)
        thread.start()
        
    def has_content(self):
        """캔버스에 내용이 있는지 확인"""
        return len(self.stroke_points) > 0
        
    def start_draw(self, event):
        """그리기 시작"""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        self.stroke_points.append((event.x, event.y))
        
    def draw_motion(self, event):
        """마우스 움직임에 따라 그리기"""
        if self.drawing:
            # 부드러운 선 그리기
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                  width=self.brush_width, fill='black',
                                  capstyle=tk.ROUND, smooth=tk.TRUE)
            
            # PIL 이미지에도 그리기 (안티앨리어싱 적용)
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                         fill='black', width=self.brush_width)
            
            # 중간 점들도 채우기 (더 부드러운 선을 위해)
            distance = ((event.x - self.last_x)**2 + (event.y - self.last_y)**2)**0.5
            if distance > 0:
                steps = int(distance / 2)
                for i in range(steps):
                    t = i / steps
                    x = int(self.last_x + t * (event.x - self.last_x))
                    y = int(self.last_y + t * (event.y - self.last_y))
                    self.draw.ellipse([x - self.brush_width//2, y - self.brush_width//2,
                                     x + self.brush_width//2, y + self.brush_width//2],
                                    fill='black')
            
            self.stroke_points.append((event.x, event.y))
            self.last_x = event.x
            self.last_y = event.y
            
    def end_draw(self, event):
        """그리기 종료"""
        self.drawing = False
        
    def clear_canvas(self):
        """캔버스 지우기"""
        self.canvas.delete("all")
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="?", fg="blue")
        self.confidence_label.config(text="신뢰도: 0%")
        self.prob_label.config(text="")
        self.stroke_points = []
        
    def preprocess_image(self):
        """개선된 이미지 전처리"""
        # 가우시안 블러로 부드럽게
        img = self.image.filter(ImageFilter.GaussianBlur(radius=1))
        
        # 이미지 반전 (MNIST는 검은 배경에 흰 글씨)
        img = ImageOps.invert(img)
        
        # 이미지에서 실제 내용이 있는 부분만 추출
        bbox = img.getbbox()
        if bbox:
            # 여백 추가
            padding = 20
            left = max(0, bbox[0] - padding)
            top = max(0, bbox[1] - padding)
            right = min(img.width, bbox[2] + padding)
            bottom = min(img.height, bbox[3] + padding)
            img = img.crop((left, top, right, bottom))
            
            # 종횡비 유지하면서 정사각형으로
            width, height = img.size
            max_dim = max(width, height)
            
            # 여백이 있는 정사각형 캔버스 생성
            square_size = int(max_dim * 1.2)  # 20% 여백
            new_img = Image.new('L', (square_size, square_size), 'black')
            
            # 중앙에 배치
            paste_x = (square_size - width) // 2
            paste_y = (square_size - height) // 2
            new_img.paste(img, (paste_x, paste_y))
            img = new_img
        
        # 20x20으로 리사이즈 (MNIST 원본과 유사하게)
        img = img.resize((20, 20), Image.Resampling.LANCZOS)
        
        # 28x28 캔버스의 중앙에 배치
        final_img = Image.new('L', (28, 28), 'black')
        final_img.paste(img, (4, 4))
        
        # numpy 배열로 변환
        img_array = np.array(final_img).astype('float32')
        
        # 질량 중심 계산 및 중앙 정렬
        img_array = self.center_image(img_array)
        
        # 정규화
        img_array = img_array / 255.0
        
        # 모델 입력 형식으로 변환
        img_array = img_array.reshape(28, 28, 1)
        
        return img_array
    
    def center_image(self, img_array):
        """이미지의 질량 중심을 계산하여 중앙 정렬"""
        # 질량 중심 계산
        cy, cx = ndimage.center_of_mass(img_array)
        
        # 이동 거리 계산
        shift_y = 14 - cy
        shift_x = 14 - cx
        
        # 이미지 이동
        shifted = ndimage.shift(img_array, [shift_y, shift_x], mode='constant', cval=0)
        
        return shifted
    
    def predict_digit(self, silent=False):
        """그려진 숫자 예측"""
        if not self.model_loaded:
            if not silent:
                self.result_label.config(text="!")
                self.prob_label.config(text="모델이 로드되지 않았습니다.")
            return
            
        # 이미지 전처리
        processed_img = self.preprocess_image()
        
        # 예측 수행
        prediction = self.classifier.model.predict(np.expand_dims(processed_img, axis=0), verbose=0)
        predicted_digit = np.argmax(prediction[0])
        confidence = prediction[0][predicted_digit] * 100
        
        # 결과 표시
        if confidence > 85:
            color = "green"
        elif confidence > 70:
            color = "orange"
        else:
            color = "red"
        self.result_label.config(text=str(predicted_digit), fg=color)
        self.confidence_label.config(text=f"신뢰도: {confidence:.1f}%")
        
        # 상위 5개 예측 결과 표시
        top_5_indices = np.argsort(prediction[0])[-5:][::-1]
        prob_text = "예측 확률:\n"
        for idx in top_5_indices:
            prob = prediction[0][idx] * 100
            if prob > 5:  # 5% 이상만 표시
                bar = "█" * int(prob / 10)
                prob_text += f"숫자 {idx}: {bar} {prob:.1f}%\n"
        self.prob_label.config(text=prob_text.strip())
        
        # 전처리된 이미지 시각화 (silent 모드가 아닐 때만)
        if not silent:
            self.show_processed_image(processed_img)
        
    def show_processed_image(self, img_array):
        """전처리된 이미지를 별도 창에 표시"""
        if hasattr(self, 'preview_window') and self.preview_window:
            try:
                self.preview_window.destroy()
            except:
                pass
        
        self.preview_window = tk.Toplevel(self.master)
        self.preview_window.title("전처리된 이미지 (28x28)")
        self.preview_window.geometry("200x250")
        
        # 이미지를 확대하여 표시
        img = Image.fromarray((img_array.reshape(28, 28) * 255).astype(np.uint8))
        img_large = img.resize((140, 140), Image.Resampling.NEAREST)
        
        # 그리드 추가 (선택사항)
        img_with_grid = self.add_grid(img_large, 5)
        
        # tkinter용으로 변환
        from PIL import ImageTk
        photo = ImageTk.PhotoImage(img_with_grid)
        
        label = Label(self.preview_window, image=photo)
        label.image = photo
        label.pack(pady=10)
        
        # 설명 추가
        info = Label(self.preview_window, text="MNIST 형식으로 전처리된 이미지",
                    font=("Arial", 10))
        info.pack()
        
        # 3초 후 자동으로 닫기
        self.preview_window.after(3000, lambda: self.close_preview_window())
    
    def add_grid(self, img, grid_size):
        """이미지에 그리드 추가 (시각화 개선)"""
        img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # 그리드 그리기
        for i in range(0, width, grid_size):
            draw.line([(i, 0), (i, height)], fill=(200, 200, 200), width=1)
        for i in range(0, height, grid_size):
            draw.line([(0, i), (width, i)], fill=(200, 200, 200), width=1)
            
        return img
    
    def close_preview_window(self):
        """미리보기 창 닫기"""
        if hasattr(self, 'preview_window') and self.preview_window:
            try:
                self.preview_window.destroy()
                self.preview_window = None
            except:
                pass

def main():
    """메인 실행 함수"""
    root = tk.Tk()
    app = ImprovedDigitDrawingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()