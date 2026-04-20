# 🚀 ResNet18 Image Classification Web App

An end-to-end image classification web application built using **ResNet18**, **PyTorch Lightning**, and **Flask**.  
Upload an image and get real-time predictions along with confidence scores and visual explanations.

---

## 🧠 Features

- 🔍 Image Classification using ResNet18
- ⚡ Built with PyTorch Lightning
- 🌐 Flask Web App for real-time inference
- 📊 Confidence Score using Softmax
- 🔥 Grad-CAM Visualization (Model Explainability)
- 🖼️ Image Upload + Preview
- 📜 Prediction History Tracking
- 🎨 Clean and Responsive UI

---

## 🏗️ Project Structure

    Image_Classification_WebApp/
    │
    ├── src/
    │   ├── app.py                 # Flask application
    │   ├── train.py               # Model training script
    │   ├── resnet18_model.pth     # Trained model weights
    │
    │   ├── templates/
    │   │   └── index.html         # Frontend UI
    │
    │   └── static/
    │       └── uploads/           # Uploaded images + Grad-CAM outputs
    │
    ├── requirements.txt       
    └── README.md

---

## ⚙️ Installation

**1. Clone the repository**

    git clone https://github.com/your-username/resnet18-image-classification-webapp.git
    cd resnet18-image-classification-webapp

**2. Create a virtual environment**

    python -m venv venv
    source venv/bin/activate

**3. Install dependencies**

    1. pip install torch torchvision pytorch-lightning flask pillow opencv-python
    2. pip install -r requirements.txt


---

## 🧪 Run the Project

    cd src
    python app.py

Open in browser: http://127.0.0.1:5000

---

## 📸 How It Works

1. Upload an image
2. Model predicts the class
3. Results include:
   - Prediction label
   - Confidence score

---

## 🧠 Model Details

| Property      | Value             |
|---------------|-------------------|
| Architecture  | ResNet18          |
| Framework     | PyTorch Lightning |
| Dataset       | CIFAR-10          |
| Loss Function | CrossEntropyLoss  |
| Optimizer     | Adam              |

---

## 🔥 Grad-CAM (Explainability)

Grad-CAM visualizes which regions of the image the model focuses on when making a prediction, improving model transparency and interpretability.

---

## 📊 Example Output

- **Prediction:** Cat  
- **Confidence:** 92.34%  

---

## 🚀 Future Improvements

- 🌍 Deploy on cloud (Render / HuggingFace Spaces)
- 📱 Mobile-friendly UI
- 🔝 Top-3 predictions display
- 🗄️ Database integration (SQLite / PostgreSQL)
- ⚡ Migrate backend to FastAPI

---

## 🧠 Learnings

- Deep Learning model training & deployment
- PyTorch Lightning workflow
- Flask backend integration
- Model interpretability with Grad-CAM
- Debugging real-world ML pipelines

---

## 👨‍💻 Author

**Anuj Pandey**