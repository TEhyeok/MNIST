#!/usr/bin/env python
# MNIST 손글씨 숫자 인식 프로그램 실행 스크립트

import os
import subprocess
import sys

def main():
    """모델 확인 후 그리기 앱 실행"""
    
    # 모델 파일 존재 확인
    if not os.path.exists('mnist_model.h5'):
        print("모델 파일이 없습니다. 먼저 모델을 훈련시킵니다...")
        print("-" * 50)
        
        # mnist_classifier.py 실행
        result = subprocess.run([sys.executable, "mnist_classifier.py"])
        
        if result.returncode != 0:
            print("모델 훈련 중 오류가 발생했습니다.")
            sys.exit(1)
            
        print("-" * 50)
        print("모델 훈련이 완료되었습니다!")
    
    # 그리기 앱 실행
    print("\n손글씨 숫자 인식 프로그램을 시작합니다...")
    subprocess.run([sys.executable, "digit_drawing_app.py"])

if __name__ == "__main__":
    main()