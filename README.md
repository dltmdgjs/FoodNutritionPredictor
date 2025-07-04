![header](https://capsule-render.vercel.app/api?type=waving&color=timeGradient&height=200&section=header&text=FoodNutritionPredictor&fontSize=40)

# Food nutrition predictor

### 목차
[1. 소개](#소개)  
[2. 주요 기능](#주요_기능)  
[3. 프로그램 실행](#프로그램_실행)  
[4. 모델 학습 과정](#모델_학습_과정)  
[5. Reference](#Reference)  
[6. License](#License)  
[7. 기타](#기타)  

### 소개
- 업로드 된 음식 사진을 사전에 학습된 Resnet50 기반의 분류 모델로 분석하여 음식 종류를 예측하고, 해당 음식의 영양 성분 정보를 제공해줍니다.

### 주요 기능
- 음식 업로드 (파일 or 카메라 촬영 선택 가능)
- 사용자 설정 (성별, 키, 몸무게, 활동성)
- 음식 분류 및 영양성분 제공
- 사용자 설정에 맞춘 하루 영양 섭취 기준 제공 및 파이 차트로 시각화
- 기준 초과 시 경고
- 일별 식사량 기록 및 일별 식사량 참조 기능 제공

### 프로그램 실행  
- 실행
```
  zip 파일 다운로드 후 압축 해제
  cd "프로젝트 폴더 위치"
  streamlit run app.py
```

- 전체 실행  
![Image](https://github.com/user-attachments/assets/8f327594-d2a4-4488-9a1b-a29634d72bc5)
```
[사용자 정보 입력(성별, 키, 몸무게, 활동성)]
  → [사용자 음식 이미지 업로드(촬영 또는 파일 업로드) 및 중량 입력] 
     → [ResNet50으로 음식 분류 (classification)]
        → [예측된 음식에 해당하는 영양 정보 추출(csv 매핑)]
           - [파이 차트 시각화 및 하루 섭취 기준 비교]
           - [식사량 저장 및 불러오기]
```
  
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
- Model : Resnet 50
- Data Set
  - 이미지 : food 101 (kaggle), 학습과 검증 셋을 9:1로 나누어 각 epoch마다 accuarcy를 측정.
  - 영양성분 정보 : food-101 nutritional information (kaggle).
- 학습환경 : google colab(jupyter notebook), A100 gpu 사용
- 학습시간 : 약 3시간
- 방법
  - 20 epoch : 첫 10개의 epoch에 대해서 레이어를 동결시켜 학습하고, 나머지 10개는 동결 해제 후 fine tuning 진행.
  - learning rate : 첫 10개의 epoch에 대해서 0.001로 설정하고, 나머지 10개는 1e-5로 더 작게 설정하여 오버피팅 방지함.
  - batch size : 32
  - optimizer : Adam
  - 손실함수 : Negative Log Likelihood Loss (NLLLoss)
- 결과
  - accuracy 비교
  <img width="800" alt="Image" src="https://github.com/user-attachments/assets/94e5e56b-2111-4dd7-8a76-60ea4c7cc97c" />

  - Epoch 1~10: 정확도 증가가 비교적 완만하고, 더 학습하면 validation accuracy가 낮아질 가능성 있음.
  - Epoch 11~20: Fine-tuning 이후 빠른 정확도 상승이 나타나며, validation accuracy는 75.95%까지 도달함.


### Reference
- [이미지 데이터 셋](https://www.kaggle.com/datasets/dansbecker/food-101/discussion?sort=hotness)
- [영양성분 정보](https://www.kaggle.com/datasets/sanadalali/food-101-nutritional-information)
- [steamlit api](https://docs.streamlit.io/develop/api-reference)
- [모델 학습 방법 관련](https://ssam2s.tistory.com/4)

### License
MIT License

### 기타
#### 사용 기술 (Tech Stack)
- Framework: Streamlit
- ML Model: ResNet50 (Image Classification)
- 학습 환경: Google Colab, PyTorch
- Data: Food-101 image, Food-101 nutrition CSV
- 시각화: matplotlib, Streamlit chart

#### 파일정보
- app.py : 프로그램 실행 파일
- Resnet50_food101.ipynb : 모델 학습 실행 파일(google colab 활용), (학습 결과 저장)
- resnet50_food101_epoch20.pth : 학습된 모델 파일
- classes.txt : 음식 클래스명 파일
- nutrition_numeric_fixed.csv : 영양성분 정보 파일
- food101_korean_mapping.xlsx : 한글 영문 매핑 파일
- meal_log.csv : 섭취 기록 파일

#### 향후 개선 계획
- 사용자 맞춤형 식단 추천 기능
- 예측된 음식에 대한 유사 음식 추천
- LLM 활용한 하루 식단 평가 기능
- 더 많은 음식 종류 (공공 데이터 셋 활용)
- 모바일 지원(IOS, Android)
