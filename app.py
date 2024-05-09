from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

class_Labels= {
    0: "apple",
    1: "banana",
    2: "blackgram",
    3: "chickpea",
    4: "coconut",
    5: "coffee",
    6: "cotton",
    7: "grapes",
    8: "jute",
    9: "kidneybeans",
    10: "lentil",
    11: "maize",
    12: "mango",
    13: "mothbeans",
    14: "mungbean",
    15: "muskmelon",
    16: "orange",
    17: "papaya",
    18: "pigeonpeas",
    19: "pomegranate",
    20: "rice",
    21: "watermelon"

               }

model = pickle.load(open('Crop_Final.pkl','rb'))

@app.route("/")   # used to define the routes on the web pages
def index_page():
    return render_template("index.html")

@app.route("/recommendation", methods = ["GET", "POST"])
def recommendation_page():
    float_features = [float(x) for x in request.form.values()]

    float_features = np.array(float_features)

    mean = [50.55181818,	53.36272727,	48.14909091,	25.61624385,	71.48177922,	6.469480065,	103.4636554]
    std_dev = [36.90894258,	32.97838509,	50.63641835,	5.062597617,	22.25875106,	0.773761773,	54.94589656]

    reverse_scaled_features = []

    # for i in range(len(float_features)):
    #     reverse_scaled_features.append((float_features[i] - mean[i])/std_dev[i])
    # features = np.array(reverse_scaled_features)

    result = model.predict([float_features])

    print("++++++")
    print(result)

    # prediction = class_Labels[result[0]]

    return render_template("index.html", y= result[0])
    

if __name__ == "__main__":
    app.run(debug=True)
