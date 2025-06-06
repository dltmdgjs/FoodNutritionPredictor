import os
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import rc

# matplotlib 한글 폰트 설정
rc('font', family='AppleGothic') 			 
plt.rcParams['axes.unicode_minus'] = False


# 한글 음식명 매핑
df_kor_map = pd.read_excel("./data/food101_korean_mapping.xlsx")
eng_to_kor = dict(zip(df_kor_map['영문 클래스명'], df_kor_map['한글 음식명']))


# 클래스 목록 로드
with open("./data/classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]


# 모델 로드
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(classes))  # Food-101 기준 클래스 수
#TODO: epoch 15~30으로 다시 학습해야함. -> 20epoch로 학습된 모델을 사용
model.load_state_dict(torch.load("./data/resnet50_food101_epoch20.pth", map_location="cpu"))
model.eval()


# 중량별 영양정보 CSV 로드
df = pd.read_csv("./data/nutrition_numeric_fixed.csv")


# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])




# 사용자 맞춤 일일 영양소 권장량 계산
def get_custom_dri(sex, age, weight, activity):
    # 기본값
    calories = 25 * weight
    protein = 0.8 * weight
    fats = 0.8 * weight
    carbs = calories * 0.55 / 4
    fiber = 25
    sodium = 2000
    sugars = 50

    if activity == "높음":
        calories *= 1.2
    elif activity == "낮음":
        calories *= 0.9

    if sex == "여성":
        protein *= 0.9
        fats *= 0.9

    if age <= 18:
        calories *= 1.1
        protein *= 1.2
    elif age >= 65:
        calories *= 0.9

    return {
        '열량': round(calories, 1),
        '탄수화물': round(carbs, 1),
        '단백질': round(protein, 1),
        '지방': round(fats, 1),
        '식이섬유': fiber,
        '당': sugars,
        '나트륨': sodium
    }


# 영양소 비율에 따른 색상 매핑
def get_color_by_ratio(ratio):
    if ratio <= 0.1:
        return "#dcdcdc"  # 회색
    elif ratio <= 0.3:
        return "#89CFF0"  # 연파랑
    elif ratio <= 0.5:
        return "#FFB347"  # 주황
    elif ratio <= 0.7:
        return "#FF6961"  # 연빨강
    else:
        return "#CC0000"  # 진한 빨강


# 파이 차트 - 영양소 섭취량 시각화
def plot_nutrient_pies(row):
    nutrient_keys = {
        'calories': '열량',
        'carbohydrates': '탄수화물',
        'protein': '단백질',
        'fats': '지방',
        'fiber': '식이섬유',
        'sugars': '당',
        'sodium': '나트륨'
    }


    DAILY_NUTRIENTS = get_custom_dri(
        st.session_state['sex'],
        st.session_state['age'],
        st.session_state['weight'],
        st.session_state['activity']
    )

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 8))  # 👈 2×2 그리드
    axes = axes.flatten() 

    for i, (key, label) in enumerate(nutrient_keys.items()):
        consumed = float(row[key])
        recommended = DAILY_NUTRIENTS[label]
        ratio = consumed / recommended
        pie_color = get_color_by_ratio(ratio)
        values = [min(consumed, recommended), max(recommended - consumed, 0)]
        pie_labels = [f"{label} 섭취량", "남은 권장량"]

        axes[i].pie(
            values,
            labels=pie_labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=[pie_color, "#eeeeee"]
        )
        axes[i].set_title(f"{label} ({consumed:.1f} / {recommended})")

        if consumed > recommended:
            st.warning(f"{label} 섭취량이 하루 권장량을 초과했습니다. ({consumed:.1f} > {recommended})")

    if len(nutrient_keys) < len(axes):
        axes[-1].axis("off")

    plt.tight_layout()
    st.pyplot(fig)



# 보간 함수 정의
def get_nutrition_for_weight(food_label, weight, df):
    subset = df[df['label'] == food_label].copy()

    # 숫자형 변환
    for col in ['weight', 'calories', 'protein', 'carbohydrates', 'fats']:
        subset[col] = pd.to_numeric(subset[col], errors='coerce')

    subset = subset.sort_values('weight')

    if subset.empty:
        return None

    # 정확히 일치
    exact = subset[subset['weight'] == weight]
    if not exact.empty:
        return exact.iloc[0]
    
  

    # 입력 중량이 최소보다 작을 때 → 최소값 기준 비례 추정
    min_row = subset.iloc[0]
    if weight < min_row['weight']:
        scale = weight / min_row['weight']
        interpolated = min_row.copy()
        for col in ['calories', 'protein', 'carbohydrates', 'fats', "fiber", "sugars", "sodium"]:
            interpolated[col] = min_row[col] * scale
        interpolated['weight'] = weight
        return interpolated
    
    # 입력 중량이 최대보다 클 때 → 최대값 기준 비례 추정
    max_row = subset.iloc[-1]
    if weight > max_row['weight']:
        scale = weight / max_row['weight']
        interpolated = max_row.copy()
        for col in ['calories', 'protein', 'carbohydrates', 'fats', "fiber", "sugars", "sodium"]:
            interpolated[col] = max_row[col] * scale
        interpolated['weight'] = weight
        return interpolated

    # 일반적인 선형 보간
    lower = subset[subset['weight'] < weight].tail(1)
    upper = subset[subset['weight'] > weight].head(1)

    if lower.empty or upper.empty:
        return None

    x0, x1 = lower['weight'].values[0], upper['weight'].values[0]
    ratio = (weight - x0) / (x1 - x0)

    interpolated = lower.iloc[0].copy()
    for col in ['calories', 'protein', 'carbohydrates', 'fats', "fiber", "sugars", "sodium"]:
        interpolated[col] = lower.iloc[0][col] + ratio * (upper.iloc[0][col] - lower.iloc[0][col])
    interpolated['weight'] = weight
    return interpolated



# 일일 섭취 기록 저장 함수
def save_meal_log(food, weight, nutrition_dict):
    log_df = pd.DataFrame([{
        "날짜": datetime.today().strftime("%Y-%m-%d"),
        "음식": food,
        "중량": weight,
        **nutrition_dict
    }])
    
    try:
        existing = pd.read_csv("./data/meal_log.csv")
        log_df = pd.concat([existing, log_df], ignore_index=True)
    except FileNotFoundError:
        pass

    log_df.to_csv("./data/meal_log.csv", index=False)



# 영양 섭취량 차트
def plot_daily_nutrition_log():
    try:
        df = pd.read_csv("./data/meal_log.csv")
        if df.empty:
            st.info("기록된 식단이 없습니다.")
            return
        
        df_grouped = df.groupby("날짜")[["열량"]].sum()
        st.bar_chart(df_grouped, color="#EFE080", x_label="날짜", y_label="열량 (kcal)")
        df_grouped = df.groupby("날짜")[["탄수화물"]].sum()
        st.bar_chart(df_grouped, color="#6ECCF7", x_label="날짜", y_label="탄수화물 (g)")
        df_grouped = df.groupby("날짜")[["단백질"]].sum()
        st.bar_chart(df_grouped, color="#FF6961", x_label="날짜", y_label="단백질 (g)")
        df_grouped = df.groupby("날짜")[["지방"]].sum()
        st.bar_chart(df_grouped, color="#CC0000", x_label="날짜", y_label="지방 (g)")
        df_grouped = df.groupby("날짜")[["식이섬유"]].sum()
        st.bar_chart(df_grouped, color="#37f03e", x_label="날짜", y_label="식이섬유 (g)")
        df_grouped = df.groupby("날짜")[["당"]].sum()
        st.bar_chart(df_grouped, color="#FFEA47", x_label="날짜", y_label="당 (g)")
        df_grouped = df.groupby("날짜")[["나트륨"]].sum()
        st.bar_chart(df_grouped, color="#C4ECFF", x_label="날짜", y_label="나트륨 (mg)")

        if st.button("식단 기록 삭제"):
            try:
                os.remove("./data/meal_log.csv")
                st.success("식단 기록이 삭제되었습니다.")
            except FileNotFoundError:
                st.warning("삭제할 식단 기록이 없습니다.")

    except FileNotFoundError:
        st.warning("아직 저장된 식단이 없습니다.")


# Streamlit UI 시각화
# 사이드바 사용자 정보 설정
with st.sidebar:
    st.header("사용자 정보 설정")
    
    if 'sex' not in st.session_state:
        st.session_state['sex'] = "남성"
    if 'age' not in st.session_state:
        st.session_state['age'] = 30
    if 'weight' not in st.session_state:
        st.session_state['weight'] = 70
    if 'activity' not in st.session_state:
        st.session_state['activity'] = "보통"

    st.session_state['sex'] = st.selectbox("성별", ["남성", "여성"], index=0)
    st.session_state['age'] = st.number_input("나이 (세)", min_value=1, max_value=120, value=30, step=1)
    st.session_state['weight'] = st.number_input("몸무게 (kg)", min_value=30, max_value=300, value=70, step=1)
    st.session_state['activity'] = st.selectbox("활동 수준", ["높음", "보통", "낮음"], index=1)
    if st.button("지난 식사량 보기"):
        plot_daily_nutrition_log()

# 메인 페이지 설정
st.title("음식 이미지 기반 영양 정보 분석")
st.write("상단의 사이드바 > 를 열어 사용자 정보를 입력해주세요.")
st.write("음식 이미지를 업로드하면 해당 음식의 영양 정보를 분석해드립니다.")

option = st.radio("이미지 업로드 방식 선택", ["파일 업로드", "카메라 촬영"])
if option == "파일 업로드":
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif option == "카메라 촬영":
    uploaded_file = st.camera_input("사진을 찍으세요")
    if uploaded_file:
        image = Image.open(uploaded_file)

weight = st.number_input("섭취 중량(g)을 입력해주세요. (최대 3000g)", min_value=10, max_value=3000, value=100, step=10)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드한 이미지", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)
    
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_idx].item()

    if confidence < 0.5:
        st.error("제공해주신 사진으로 정확하게 예측되지 않습니다. 다른 음식 사진을 올려주세요.")
        st.write(f"예측 신뢰도: {confidence:.1%}")
    else:
        predicted_class = classes[predicted_idx]
        translated_name = eng_to_kor.get(predicted_class, predicted_class)
        st.subheader(f"예측된 음식: {translated_name} ({predicted_class})")
        st.write(f"예측 신뢰도: {confidence:.1%}")

        row = get_nutrition_for_weight(predicted_class, weight, df)
        if row is not None:
            st.markdown(f"### {translated_name} ({int(row['weight'])}g) 기준 영양정보")
            st.write(f"- **열량**: {row['calories']:.1f} kcal")
            st.write(f"- **탄수화물**: {row['carbohydrates']:.1f} g")
            st.write(f"- **단백질**: {row['protein']:.1f} g")
            st.write(f"- **지방**: {row['fats']:.1f} g")
            st.write(f"- **식이섬유**: {row['fiber']:.1f} g")
            st.write(f"- **당**: {row['sugars']:.1f} g")
            st.write(f"- **나트륨**: {row['sodium']:.1f} mg")
            st.subheader("하루 권장 섭취량 대비")
            st.write("권장 섭취량은 사이드바에서 설정한 사용자 정보에 따라 달라집니다.")
            plot_nutrient_pies(row)
            if st.button("식사 기록하기"):
                save_meal_log(translated_name, weight, {
                    '열량': row['calories'],
                    '탄수화물': row['carbohydrates'],
                    '단백질': row['protein'],
                    '지방': row['fats'],
                    '식이섬유': row['fiber'],
                    '당': row['sugars'],
                    '나트륨': row['sodium']
                })
                st.success("식단 기록이 저장되었습니다.")
        else:
            st.warning("해당 음식에 대한 영양정보를 찾을 수 없습니다.")