import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# --- 1. SETTINGS & CUSTOM CSS ---
st.set_page_config(page_title="AI Object Counter Pro", layout="wide", page_icon="📊")

# แต่งหน้าตาเว็บด้วย CSS
st.markdown("""
    <style>
    /* เปลี่ยนสีพื้นหลังและฟอนต์ */
    .main {
        background-color: #0f1116;
    }
    .stMetric {
        background-color: #1c1f26;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #2d3139;
    }
    h1, h2, h3 {
        color: #00f2ff !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stAlert {
        border-radius: 10px;
    }
    /* ปรับแต่ง Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161920;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    # ใช้ YOLOv8n (Nano) เพื่อความเร็วบน RX 6500 XT
    return YOLO('yolov8n.pt') 

model = load_model()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Configuration")
    st.info("ระบบวิเคราะห์วัตถุด้วย AI ความเร็วสูง")
    uploaded_file = st.file_uploader("📂 อัปโหลดรูปภาพ (JPG, PNG)", type=["jpg", "jpeg", "png"])
    

# --- 4. MAIN CONTENT ---
st.title("📊 AI Object Counter & Vision Analytics")
st.write("---")

if uploaded_file is not None:
    # อ่านรูปภาพ
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # แบ่งส่วนการแสดงผล
    col_img, col_stat = st.columns([2, 1])

    with st.spinner('🚀 AI กำลังประมวลผลผ่าน Neural Network...'):
        # ตรวจจับ
        results = model(img_array)
        
        # วาดกรอบและดึงผลลัพธ์
        res_plotted = results[0].plot()
        
        # นับจำนวน
        counts = results[0].boxes.cls.tolist()
        names = results[0].names
        summary = {}
        for c in counts:
            obj_name = names[int(c)]
            summary[obj_name] = summary.get(obj_name, 0) + 1

    # แสดงผลรูปภาพ
    with col_img:
        st.subheader("🖼️ Vision Detection Result")
        st.image(res_plotted, use_container_width=True, caption="AI-Identified Objects")

    # แสดงสถิติและข้อมูลวิเคราะห์
    with col_stat:
        st.subheader("📈 Analytics Summary")
        
        # โชว์ Metric รวม
        total_objects = len(counts)
        st.metric(label="Total Objects Found", value=total_objects)
        
        st.write("---")
        
        # แสดงรายการที่นับได้
        if summary:
            for obj, count in summary.items():
                st.write(f"🔹 **{obj.capitalize()}**: `{count}` items")
            
            st.divider()
            # กราฟแท่งสรุปผล 
            st.bar_chart(summary)
        else:
            st.warning("ไม่พบวัตถุที่ระบุได้ในภาพนี้")

else:
    # หน้าต้อนรับเวลาไม่มีการอัปโหลดไฟล์
    st.empty()
    col_mid = st.columns([1, 2, 1])[1]
    with col_mid:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=150)
        st.info("กรุณาอัปโหลดรูปภาพที่ Sidebar ด้านซ้าย เพื่อเริ่มการวิเคราะห์")

# --- 5. FOOTER ---
st.divider()
st.caption("This project was made for studying")