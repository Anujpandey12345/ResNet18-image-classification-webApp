from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
# Load model
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import uuid, os

app = Flask(__name__)

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

classes = ["airplane","car","bird","cat","deer","dog","frog","horse","ship","truck"]


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
    return render_template("index.html", prediction=prediction, confidence=confidence, image_path=image_path)
if __name__ == "__main__":
    app.run(debug=True)