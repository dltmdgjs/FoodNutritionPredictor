![header](https://capsule-render.vercel.app/api?type=waving&color=timeGradient&height=200&section=header&text=FoodNutritionPredictor&fontSize=40)

# Food nutrition predictor

### 목차
[1. 소개](#소개)  
[2. 주요 기능](#주요_기능)  
[3. 프로그램 실행](#프로그램_실행)  
[4. 모델 학습 과정](#모델_학습_과정)  
[5. Reference](#Reference)  
[6. License](#License)  
[7. 기타(참고사항)](#기타(참고사항))  

### 소개
- 업로드 된 음식 사진을 사전에 학습된 Resnet50 모델로 예측하여 해당 음식의 영양성분을 분석해줍니다.

### 주요 기능
- 음식 업로드 (파일 or 카메라 촬영 선택 가능)
- 사용자 설정 (성별, 키, 몸무게, 활동성)
- 음식 예측 및 영양성분 제공
- 사용자 설정에 맞춘 하루 영양 섭취 기준 제공 및 파이 차트로 시각화
- 일별 식사량 기록 및 일별 식사량 참조 기능 제공

### 프로그램 실행
![Image](https://github.com/user-attachments/assets/8f327594-d2a4-4488-9a1b-a29634d72bc5)  
  
- 첫 화면
  - 사이드 바: 사용자 정보 입력란, 지난 식사량 보기 버튼
  - 메인 화면: 이미지 업로드, 중량 입력  
<img width="800" alt="Image" src="https://github.com/user-attachments/assets/70d0d606-373b-4b5f-a3bc-72698d0d725c" />

  
- 이미지 업로드
  - 업로드한 이미지 화면에 표시
<img width="800" alt="Image" src="https://github.com/user-attachments/assets/8940d322-b15c-487c-97cd-a7c2b0a6a0f4" />
  

- 예측
  - 예측 정확도 밎 중량별 영양성분 정보 제공
<img width="800" alt="Image" src="https://github.com/user-attachments/assets/c66e8724-b53e-4d84-b01c-a69198a6d459" />
 

- 일일 권장량 비교
  - 일일 영양 성분 비율에 따른 차등 색상  
<img width="800" alt="Image" src="https://github.com/user-attachments/assets/59c0a46c-5d82-45a2-9cdd-1c9d2d198fa3" />
  

- 지난 식사량
  - 영양 성분 별 일별 총 섭취량 정보 제공 
<img width="800" alt="Image" src="https://github.com/user-attachments/assets/3abd9567-f41c-4718-bc86-304c3f7e2776" />

 


### 모델 학습 과정

### Reference

### License

### 기타(참고사항)
