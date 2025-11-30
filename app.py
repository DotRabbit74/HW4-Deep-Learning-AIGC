import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 頁面基本設定 ---
st.set_page_config(page_title="浣熊 vs 狸貓", page_icon="🦝")
st.title("🦝 浣熊 vs 狸貓 AI 辨識器")
st.write("狀態：準備就緒，請上傳圖片。")

# --- 參數設定 ---
MODEL_PATH = 'raccoon_tanuki_model.pth'
CONFIDENCE_THRESHOLD = 0.6 

# --- 載入模型 (移除 Cache 以確保穩定性) ---
def get_model():
    device = torch.device("cpu")
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        class_names = checkpoint['classes']
        
        # 建立模型
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, class_names, device
    except:
        return None, None, None

# --- 影像處理 ---
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- Grad-CAM ---
def get_gradcam(model, input_tensor, original_image):
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    img_resized = np.array(original_image.resize((224, 224))) / 255.0
    return show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)

# --- 主程式邏輯 ---
# 1. 先顯示上傳按鈕 (確保這個 UI 永遠存在)
uploaded_file = st.file_uploader("📷 請選擇一張 JPG 或 PNG 圖片", type=["jpg", "jpeg", "png"])

# 2. 如果有上傳，才開始載入模型並預測
if uploaded_file is not None:
    st.write("🔄 正在分析中...")
    
    # 載入模型
    model, class_names, device = get_model()
    
    if model is None:
        st.error("找不到模型檔案，請檢查 GitHub 或 Colab 檔案區。")
    else:
        try:
            # 讀圖與預測
            image = Image.open(uploaded_file).convert('RGB')
            input_tensor = process_image(image)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            top_prob, top_idx = torch.max(probs, 0)
            top_class = class_names[top_idx]
            top_prob_val = top_prob.item()
            
            # --- 顯示結果區 ---
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption='你的圖片', use_column_width=True)
            
            with col2:
                st.subheader("📊 分析結果")
                
                # OOD 判斷
                is_ood = False
                if top_class == 'other':
                    is_ood = True
                    st.error("🚫 結果：以上皆非 (Other)")
                elif top_prob_val < CONFIDENCE_THRESHOLD:
                    is_ood = True
                    st.warning(f"🤔 結果：不確定 (似 {top_class}?)")
                    st.write(f"信心度 {top_prob_val*100:.1f}% 太低。")
                else:
                    if top_class == 'raccoon':
                        st.success("🦝 結果：浣熊 (Raccoon)")
                    elif top_class == 'tanuki':
                        st.info("🍂 結果：狸貓 (Tanuki)")
                    st.metric("信心度", f"{top_prob_val*100:.1f}%")

                st.bar_chart({name: float(p) for name, p in zip(class_names, probs)})

            # --- 進階功能 (熱點圖 + 教學) ---
            if not is_ood:
                st.markdown("---")
                st.subheader("🔥 AI 視覺熱點")
                cam_vis = get_gradcam(model, input_tensor, image)
                st.image(cam_vis, caption='紅色區域為判斷依據', width=350)
                
                st.markdown("---")
                st.subheader("🎓 特徵比一比")
                
                # 樣式設定
                style_rac = "border:2px solid #4CAF50; background:#e8f5e9" if top_class == 'raccoon' else "opacity:0.5"
                style_tan = "border:2px solid #FF9800; background:#fff3e0" if top_class == 'tanuki' else "opacity:0.5"
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""<div style="{style_rac}; padding:10px; border-radius:10px">
                    <h4 style="color:#2E7D32; text-align:center">🦝 浣熊特徵</h4>
                    <ul><li><b>尾巴有環紋</b></li><li>五指分開 (像手)</li><li>眼罩分開</li></ul></div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div style="{style_tan}; padding:10px; border-radius:10px">
                    <h4 style="color:#EF6C00; text-align:center">🍂 狸貓特徵</h4>
                    <ul><li><b>尾巴無環紋</b></li><li>腳掌像狗肉墊</li><li>眼罩相連</li></ul></div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"發生錯誤: {e}")
