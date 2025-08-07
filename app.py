import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)
#Loading the model
#CREATING PYTORCH MODEL

class ANN_Model(nn.Module):
  def __init__(self, input_features=8, hidden1=20, hidden2=20, out_features=2):
    super().__init__()
    self.f_connected1 = nn.Linear(input_features,hidden1)
    self.f_connected2 = nn.Linear(hidden1, hidden2)
    self.out = nn.Linear(hidden2,out_features)

  def forward(self,x):
    x = F.relu(self.f_connected1(x))
    x = F.relu(self.f_connected2(x))
    x = self.out(x)
    return x
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print((torch.FloatTensor(list(data.values()))))
    new_data = torch.FloatTensor(list(data.values()))
    with torch.no_grad():
        output = model(new_data).argmax().item()
    return jsonify(output)

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = torch.FloatTensor(data)
    print(final_input)
    with torch.no_grad():
        output = model(final_input).argmax().item()
    return render_template("home.html", prediction_text="Your diabetes test result is {}".format(output))
        
if __name__ == "__main__":
    app.run(debug=True)