# -- coding: utf-8
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from flask import Flask, jsonify, request
from BlurryModel import BlurryModel

app = Flask(__name__)
svm = BlurryModel('minMaxScaler.pkl', 'pca_4.pkl', 'svm.joblib')


@app.route('/hello', methods=['GET'])
def helloworld():
    data = {"data": "Hello World"}
    return jsonify(data)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == 'GET':
        pandas_data = [[request.args.get('mean_lap'), request.args.get('std_lap'), request.args.get('max_lap'),
                        request.args.get('mean_sob'), request.args.get('std_sob'), request.args.get('max_sob'),
                        request.args.get('mean_sch'), request.args.get('std_sch'), request.args.get('max_sch')]]
        result = svm.predict(pandas_data)
        data = {"label": result}
        return jsonify(data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
