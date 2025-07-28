# 사용자 데이터 수집을 위한 GUI 애플리케이션
import tkinter as tk
from tkinter import Canvas, Button, Label, Frame, Scale, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import cv2
from scipy import ndimage
import os

class DataCollectionApp:
    
    def __init__(self, window):
        self.window = window
        self.window.title("MNIST 사용자 데이터 수집기")
        self.window.configure(bg='#f0f0f0')
        
        # 데이터 저장 파일 경로
        self.images_file = 'user_data_images.npy'
        self.labels_file = 'user_data_labels.npy'
        
        # 수집된 데이터
        self.collected_images = []
        self.collected_labels = []
        self.target_count = 200
        
        # 현재 선택된 라벨
        self.current_label = tk.StringVar(value="?")
        
        # 그리기 상태 변수
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
        
        # 캔버스 크기
        self.canvas_width = 280
        self.canvas_height = 280
        
        # PIL 이미지 (실제 그리기용)
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        # 기존 데이터 로드
        self.load_existing_data()
        
        # GUI 구성
        self.setup_gui()
        
        # 진행상황 업데이트
        self.update_progress()
        
    def load_existing_data(self):
        """기존에 수집된 데이터가 있으면 로드"""
        try:
            if os.path.exists(self.images_file) and os.path.exists(self.labels_file):
                images = np.load(self.images_file)
                labels = np.load(self.labels_file)
                self.collected_images = images.tolist()
                self.collected_labels = labels.tolist()
                print(f"기존 데이터 {len(self.collected_images)}개를 로드했습니다.")
        except Exception as e:
            print(f"기존 데이터 로드 중 오류: {e}")
            self.collected_images = []
            self.collected_labels = []
            
    def setup_gui(self):
        """GUI 컴포넌트 설정"""
        # 제목 라벨
        title_label = Label(self.window, text="사용자 데이터 수집기", 
                          font=("Arial", 20, "bold"), bg='#f0f0f0', fg='black')
        title_label.pack(pady=10)
        
        # 진행상황 표시
        self.progress_label = Label(self.window, text="", 
                                  font=("Arial", 14), bg='#f0f0f0', fg='blue')
        self.progress_label.pack(pady=5)
        
        # 설명 라벨
        instruction_label = Label(self.window, text="1. 아래 캔버스에 숫자를 그리세요\n2. 올바른 숫자 버튼을 누르세요\n3. '저장' 버튼을 눌러 데이터를 저장하세요", 
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
        
        # 라벨 선택 프레임
        label_frame = Frame(self.window, bg='#f0f0f0')
        label_frame.pack(pady=10)
        
        Label(label_frame, text="이 숫자는:", font=("Arial", 14), bg='#f0f0f0', fg='black').pack()
        
        # 0-9 버튼들
        button_frame = Frame(label_frame, bg='#f0f0f0')
        button_frame.pack(pady=5)
        
        for i in range(10):
            btn = Button(button_frame, text=str(i), font=("Arial", 12, "bold"),
                        width=3, height=1, bg='#e0e0e0', fg='black',
                        command=lambda x=i: self.select_label(x))
            btn.pack(side=tk.LEFT, padx=2)
        
        # 선택된 라벨 표시
        self.selected_label = Label(label_frame, textvariable=self.current_label, 
                                  font=("Arial", 24, "bold"), fg='red', bg='#f0f0f0')
        self.selected_label.pack(pady=5)
        
        # 버튼 프레임
        button_control_frame = Frame(self.window, bg='#f0f0f0')
        button_control_frame.pack(pady=10)
        
        # 지우기 버튼
        clear_button = Button(button_control_frame, text="지우기", command=self.clear_canvas,
                         bg="#ff6b6b", fg="black", relief=tk.RAISED, bd=3,
                         activebackground="#ff5252", activeforeground="black")
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # 저장 버튼
        self.save_button = Button(button_control_frame, text="저장", command=self.save_data,
                           bg="#4CAF50", fg="black", relief=tk.RAISED, bd=3,
                           activebackground="#45a049", activeforeground="black")
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # 완료 버튼
        finish_button = Button(button_control_frame, text="수집 완료", command=self.finish_collection,
                             bg="#2196F3", fg="black", relief=tk.RAISED, bd=3)
        finish_button.pack(side=tk.LEFT, padx=5)
        
        # 브러시 크기 조절
        brush_frame = Frame(self.window, bg='#f0f0f0')
        brush_frame.pack(pady=5)
        
        Label(brush_frame, text="브러시 크기:", bg='#f0f0f0', fg='black').pack(side=tk.LEFT)
        self.brush_size = Scale(brush_frame, from_=5, to=20, orient=tk.HORIZONTAL,
                              bg='#f0f0f0', troughcolor='#d0d0d0', fg='black')
        self.brush_size.set(12)
        self.brush_size.pack(side=tk.LEFT)
        
    def select_label(self, label):
        """라벨 선택"""
        self.current_label.set(str(label))
        
    def update_progress(self):
        """진행상황 업데이트"""
        current_count = len(self.collected_images)
        self.progress_label.config(text=f"수집된 데이터: {current_count} / {self.target_count}")
        
        if current_count >= self.target_count:
            self.progress_label.config(fg='green')
        else:
            self.progress_label.config(fg='blue')
        
    def clear_canvas(self):
        """캔버스와 이미지 초기화"""
        self.canvas.delete("all")
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.current_label.set("?")
        
    def start_drawing(self, event):
        """그리기 시작"""
        self.is_drawing = True
        self.last_x = event.x
        self.last_y = event.y
        
    def draw_line(self, event):
        """선 그리기"""
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
                    
                    # PIL 이미지에도 그리기
                    self.draw.ellipse([
                        x - brush_size//2, y - brush_size//2,
                        x + brush_size//2, y + brush_size//2
                    ], fill='black')
            
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
        
        return smoothed
    
    def save_data(self):
        """현재 그린 이미지와 라벨을 데이터에 추가"""
        # 라벨이 선택되었는지 확인
        if self.current_label.get() == "?":
            messagebox.showwarning("경고", "먼저 숫자 라벨을 선택해주세요!")
            return
        
        # 이미지 전처리
        processed_image = self.preprocess_image()
        if processed_image is None:
            messagebox.showwarning("경고", "먼저 숫자를 그려주세요!")
            return
        
        # 데이터 추가
        self.collected_images.append(processed_image)
        self.collected_labels.append(int(self.current_label.get()))
        
        # 파일로 저장
        try:
            np.save(self.images_file, np.array(self.collected_images))
            np.save(self.labels_file, np.array(self.collected_labels))
            print(f"데이터 저장 완료: {len(self.collected_images)}개")
        except Exception as e:
            messagebox.showerror("오류", f"데이터 저장 중 오류가 발생했습니다: {e}")
            return
        
        # UI 초기화
        self.clear_canvas()
        self.update_progress()
        
        # 목표 달성 시 알림
        if len(self.collected_images) >= self.target_count:
            messagebox.showinfo("완료", f"목표 {self.target_count}개 데이터 수집이 완료되었습니다!")
    
    def finish_collection(self):
        """데이터 수집 완료"""
        if len(self.collected_images) == 0:
            messagebox.showwarning("경고", "수집된 데이터가 없습니다!")
            return
            
        result = messagebox.askyesno("완료", 
                                   f"총 {len(self.collected_images)}개의 데이터가 수집되었습니다.\n"
                                   f"데이터 수집을 완료하시겠습니까?")
        if result:
            print(f"데이터 수집 완료: {len(self.collected_images)}개")
            print(f"저장된 파일: {self.images_file}, {self.labels_file}")
            self.window.quit()

def main():
    """메인 실행 함수"""
    root = tk.Tk()
    app = DataCollectionApp(root)
    root.mainloop()
    
if __name__ == "__main__":
    main()