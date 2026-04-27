from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
# Load model
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import uuid, os, json

app = Flask(__name__)


HISTORY_FILE = "history.json"

def save_history(image_path, prediction, confidence):
    data = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try:
                data = json.load(f)
            except:
                data = []

    data.append({
        "image": image_path,
        "prediction": prediction,
        "confidence": confidence
    })
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=4)


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try:
                return json.load(f)
            except:
                return []
    return []


model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)

# Load checkpoint
state_dict = torch.load("resnet18_model.pth", map_location="cpu")
# FIX: remove "model." prefix
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("model.", "")
    new_state_dict[new_key] = v

# Load cleaned weights
model.load_state_dict(new_state_dict)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

classes = ["airplane","car","bird","cat","deer","dog","frog","horse","ship","truck", "kangaroo"]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None


    if request.method == "POST":
        file = request.files["file"]

        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join("static/uploads", filename)
        file.save(filepath)

        img = Image.open(file).convert("RGB")   # ensure 3 channels
        img = transform(img)                   # convert to tensor
        img = img.unsqueeze(0)    


        with torch.no_grad():
            output = model(img)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)
        prediction = classes[pred.item()]
        confidence = round(conf.item() * 100, 2)
        image_path = filepath
        save_history(image_path, prediction, confidence)
    history = load_history()
    return render_template("index.html", prediction=prediction, confidence=confidence, image_path=image_path, history=history)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))