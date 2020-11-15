from flask import Flask, render_template, url_for, request, redirect, jsonify
from main import run, Model, single_predict, torch


app = Flask(__name__)



@app.route("/", methods=['POST','GET'])
def index():
    if (request.method == "POST"):
        data = request.form['content']
        # run()
        
        try:
            numbers = list(map(lambda x: float(x),  data.split(',')))
            output = single_predict(model, numbers)
        except:
            
            output = "You have to write 4 numbers separated by commas"
            
        return render_template('index.html', value = output)
    else:
        return render_template('index.html')







if __name__ == '__main__':
    model = Model()
    model.load_state_dict(torch.load('models/model_v1'))
    app.run(debug=True)