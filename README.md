🦝 浣熊 vs 狸貓 AI 辨識器 (ResNet18 + Grad-CAM 可解釋性)
這是一個深度學習專題實作，旨在建立一個能準確區分浣熊 (Raccoon) 與 狸貓 (Tanuki/Raccoon Dog) 的圖像分類器。由於這兩種動物外型相似，容易被大眾混淆，本專案屬於具有挑戰性的細粒度分類 (Fine-grained Classification) 任務。
專案已成功部署於 Streamlit Cloud 上，提供一個友善的網頁應用介面。
🌟 核心功能 (Features)
本分類器除了基礎的圖像辨識功能外，還加入了進階的解釋性 AI (XAI) 與可靠性判斷：
1. 三類分類
    ◦ 辨識結果包括：浣熊 (Raccoon)、狸貓 (Tanuki)。
    ◦ 以上皆非 (Out-of-Distribution, OOD) 檢測：若圖片非上述兩種動物，或模型信心度低於 60% (門檻值為 0.6)，則輸出「以上皆非」或「不確定」。
2. 可解釋性 AI (Grad-CAM)
    ◦ 輸出 Grad-CAM 熱點圖，顯示模型是看著圖片中的「哪些區域」（例如尾巴或眼部）做出判斷的。這有助於驗證模型的決策邏輯。
3. 互動式特徵對比教學
    ◦ 在結果頁面提供浣熊與狸貓的關鍵特徵對比表。
    ◦ 例如：浣熊的尾巴有黑白環狀條紋、手掌像人手；狸貓的尾巴無條紋、腳掌像狗的肉墊。
💻 技術棧 (Technology Stack)
類別
技術/模型
說明
框架
PyTorch, torchvision
深度學習核心框架。
模型
ResNet18
使用遷移學習 (Transfer Learning) 進行訓練，基於預訓練模型微調以提高準確率並減少所需訓練資料量。
解釋性
Grad-CAM (Gradient-weighted Class Activation Mapping)
用於生成模型判斷依據的熱點圖。
部署
Streamlit
快速建立互動式網頁應用程式 (App)。
環境
Streamlit Cloud, Python 3.10
雲端部署環境。
📦 專案檔案結構 (Project Structure for Streamlit Deployment)
要成功將此專案部署到 Streamlit Cloud，專案根目錄必須包含以下檔案：
my-raccoon-classifier/
├── app.py                     # Streamlit 應用程式主程式 [16]
├── requirements.txt           # Python 套件依賴清單 [16, 18]
├── raccoon_tanuki_model.pth   # 訓練好的 ResNet18 模型檔案 (約 45MB) [11, 16, 19]
├── .python-version            # 確保使用 Python 3.10 穩定版本 [17]
└── packages.txt               # 部署 Linux 底層依賴 (解決 OpenCV 錯誤) [20]
檔案內容詳情
1. requirements.txt (Python 套件清單)
此清單已經過優化，將 opencv-python-headless 提前並確保包含所有必要的深度學習函式庫。
streamlit
opencv-python-headless
torch
torchvision
grad-cam
pillow
numpy
2. .python-version
為了避免 PyTorch/OpenCV 在較新的 Python 3.13 環境中出現相容性錯誤，需強制使用 3.10 版本。
3.10
3. packages.txt
此文件告知 Streamlit Cloud 伺服器安裝必要的 Linux 系統底層函式庫，以解決 cv2 (OpenCV) 匯入時可能發生的圖形介面錯誤 (例如 libGL.so.1 錯誤)。
libgl1
libglib2.0-0
🛠️ 部署指南 (Deployment Guide)
1. 模型準備 (Colab 端)
    ◦ 在 Google Colab 上傳資料集 (dataset/train 和 dataset/val 資料夾)。
    ◦ 執行訓練程式碼 (使用 GPU 加速，訓練約 1-2 分鐘)。
    ◦ 訓練完成後，下載 raccoon_tanuki_model.pth 模型檔案到本地電腦。
2. GitHub 設置
    ◦ 建立一個新的 GitHub Repository。
    ◦ 將以下四個檔案上傳到 Repository 的根目錄：app.py、requirements.txt、.python-version、packages.txt，以及模型檔案 raccoon_tanuki_model.pth。
    ◦ (註：ResNet18 模型檔小於 100MB，可以直接使用 git push 上傳)。
3. Streamlit Cloud 部署
    ◦ 登入 Streamlit Cloud。
    ◦ 選擇「New app」，連結您的 GitHub Repository。
    ◦ 設定 Main file path 為 app.py。
    ◦ 點擊 Deploy。
部署完成後，Streamlit Cloud 會自動讀取 requirements.txt 和 packages.txt 來配置環境，並啟動您的 AI 辨識器。
🧪 資料與訓練簡介
• 資料集構成：建議每類約 100 ~ 150 張圖片 (浣熊、狸貓各約 120 張；其他類約 150~200 張，包含貓、狗、小熊貓、風景等)。
• 資料清洗提醒：狸貓 (Raccoon Dog) 資料較少，且容易混入浣熊圖片，必須人工檢查確保狸貓照片尾巴沒有環狀條紋。
• 訓練優化：訓練程式碼使用了資料增強 (transforms.RandomHorizontalFlip())，以變相擴大資料集，提升模型健壯性。
