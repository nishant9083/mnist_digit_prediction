from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Define your model class (same as in training)
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.view(-1, input_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load the trained model
input_size = 28 * 28
hidden_size = 128
output_size = 10
model = SimpleNN(input_size, hidden_size, output_size)
print("Loading model...")
with open("simple_nn_model.pkl", "rb") as f:
    model.load_state_dict(pickle.load(f, errors='ignore'))
print("Model loaded successfully!")
model.eval()

# Define a transform for preprocessing the input image
transform = transforms.Compose([
    transforms.Grayscale(),  # Ensures image is grayscale
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get image file from request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Process the image
    img = Image.open(io.BytesIO(file.read()))
    img = transform(img).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        digit = predicted.item()
    
    return jsonify({'prediction': digit})

if __name__ == '__main__':
    app.run(debug=True)
