import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ================= 1. CẤU HÌNH GIAO DIỆN =================
st.set_page_config(page_title="AI Bóng Đá Pro MAX", layout="wide", page_icon="⚽")
st.title("⚽ Hệ Thống AI Dự Đoán Ngoại Hạng Anh (Pro MAX)")
st.markdown("Sử dụng **Random Forest Classifier** kết hợp chỉ số chuyên sâu (xG, Possession) để dự đoán kết quả.")

# ================= 2. ĐỌC DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH =================
@st.cache_data
def load_data():
    df = pd.read_csv("ngoaihanganh_2024.csv")
    df = df.dropna()
    return df

features = ["Home_Form", "Away_Form", "Home_xG", "Away_xG", "Home_Poss", "Away_Poss"]
model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)

try:
    data = load_data()
    X = data[features]
    y = data["Result"]
    model.fit(X, y)
    st.success("✅ Đã kết nối thành công với bộ dữ liệu Ngoại Hạng Anh thực tế!")
except FileNotFoundError:
    st.warning("⚠️ Chưa tìm thấy file 'ngoaihanganh_2024.csv'. Hệ thống đang chạy bằng dữ liệu AI tự tạo (Mock Data).")
    np.random.seed(42)
    data_size = 1000
    mock_data = pd.DataFrame({
        "Home_Form": np.random.uniform(3, 10, data_size),
        "Away_Form": np.random.uniform(3, 10, data_size),
        "Home_xG": np.random.uniform(0.8, 3.0, data_size),
        "Away_xG": np.random.uniform(0.8, 3.0, data_size),
        "Home_Poss": np.random.uniform(30, 70, data_size),
    })
    # Tự động tính Away_Poss cho Mock Data
    mock_data["Away_Poss"] = 100 - mock_data["Home_Poss"]
    
    power_home = mock_data['Home_Form']*0.3 + mock_data['Home_xG']*3 + mock_data['Home_Poss']*0.1 + 0.5
    power_away = mock_data['Away_Form']*0.3 + mock_data['Away_xG']*3 + mock_data['Away_Poss']*0.1
    conditions = [(power_home > power_away + 1.5), (power_home < power_away - 1.0)]
    mock_data['Result'] = np.select(conditions, ['Thắng', 'Thua'], default='Hòa')
    
    X = mock_data[features]
    y = mock_data["Result"]
    model.fit(X, y)

# ================= 3. GIAO DIỆN NHẬP LIỆU =================
teams = ["Arsenal", "Man City", "Liverpool", "Chelsea", "Man Utd", "Tottenham", "Aston Villa", "Newcastle"]

st.write("### ⚙️ Thiết Lập Thông Số Trước Trận")
col1, col2 = st.columns(2)

with col1:
    st.info("🏠 ĐỘI CHỦ NHÀ")
    home_team = st.selectbox("Chọn đội chủ nhà", teams, index=0)
    home_form = st.slider(f"Phong độ {home_team} (1-10)", 1.0, 10.0, 8.5)
    home_xg = st.slider(f"xG/trận ({home_team})", 0.5, 3.5, 2.1)
    # Gom thanh trượt Possession về 1 chỗ
    home_poss = st.slider(f"Kiểm soát bóng ({home_team}) %", 20, 80, 55)

with col2:
    st.warning("✈️ ĐỘI KHÁCH")
    away_team = st.selectbox("Chọn đội khách", teams, index=1)
    away_form = st.slider(f"Phong độ {away_team} (1-10)", 1.0, 10.0, 7.0)
    away_xg = st.slider(f"xG/trận ({away_team})", 0.5, 3.5, 1.8)
    # Tự động tính tỷ lệ của đội khách
    away_poss = 100 - home_poss
    st.markdown(f"**Kiểm soát bóng ({away_team}):** `{away_poss}%` *(Tự động tính)*")

if home_team == away_team:
    st.error("⚠️ Vui lòng chọn hai đội khác nhau!")
    st.stop()

# ================= 4. HIỂN THỊ KẾT QUẢ THEO TABS =================
st.write("---")
tab1, tab2, tab3 = st.tabs(["🚀 Dự Đoán AI", "📊 So Sánh Đội Hình", "🧠 AI Explainable (Giải thích)"])

# Đưa dữ liệu về dạng Bảng (DataFrame) để tránh lỗi Warning của Scikit-learn
input_df = pd.DataFrame([[home_form, away_form, home_xg, away_xg, home_poss, away_poss]], columns=features)

with tab1:
    if st.button("🔮 Phân Tích Trận Đấu", use_container_width=True):
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_ 
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("### 🏆 Kết Quả Trận Đấu")
            if prediction == "Thắng":
                st.success(f"Dự đoán: **{home_team}** có tỷ lệ chiến thắng cao nhất!")
            elif prediction == "Thua":
                st.error(f"Dự đoán: **{away_team}** có khả năng tạo địa chấn trên sân khách!")
            else:
                st.warning("Dự đoán: Kịch bản dễ xảy ra nhất là hai đội **Hòa** nhau.")
                
        with c2:
            st.write("### 🎲 Xác Suất Phân Bổ")
            prob_df = pd.DataFrame({"Kết quả": classes, "Tỷ lệ (%)": np.round(probabilities * 100, 2)})
            st.bar_chart(prob_df.set_index("Kết quả"), color="#1f77b4")

with tab2:
    st.write("### 🕸️ Biểu Đồ Sức Mạnh Cạnh Tranh")
    labels = ['Phong độ', 'Bàn thắng kỳ vọng (xG)', 'Kiểm soát bóng (%)']
    # Chuẩn hóa dữ liệu để vẽ biểu đồ Radar đẹp hơn
    home_stats = [home_form*10, home_xg*30, home_poss]
    away_stats = [away_form*10, away_xg*30, away_poss]
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    home_stats += home_stats[:1]
    away_stats += away_stats[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, home_stats, color='blue', linewidth=2, label=home_team)
    ax.fill(angles, home_stats, color='blue', alpha=0.25)
    ax.plot(angles, away_stats, color='red', linewidth=2, label=away_team)
    ax.fill(angles, away_stats, color='red', alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig)

with tab3:
    st.write("### 🔍 Mức Độ Quan Trọng Của Các Yếu Tố (Feature Importance)")
    st.markdown("Thuật toán **Random Forest** tự động đánh giá xem yếu tố nào thực sự quyết định trận đấu:")
    
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Yếu tố": features, "Mức độ ảnh hưởng": importances}).sort_values(by="Mức độ ảnh hưởng", ascending=True)
    
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.barh(feat_df["Yếu tố"], feat_df["Mức độ ảnh hưởng"], color="teal")
    ax2.set_xlabel("Trọng số quyết định")
    st.pyplot(fig2)
