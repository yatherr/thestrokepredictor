from flask import Flask, render_template, request
import pickle


app = Flask(__name__, template_folder='templates')

# Init the path to get to our Random Forest Classifier Model
path_to_model = 'rfc.pkl'

# Load in our rfc model into the model variable
with open(path_to_model, 'rb') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def main():
    prediction = 'No Record Yet'
    if request.method == 'GET':
        # Just render the initial form, to get input
        return render_template('index.html')

    if request.method == 'POST':
        # Get the input from the user.
        age = request.form.get('age')
        bmi = request.form.get('bmi')
        glucose = request.form.get('glucose')
        hypertension = request.form.get('hypertension1')
        heart_disease = request.form.get('Disease1')

        # Convert the hypertension and heart disease inputs from a string into a 0 or 1
        if hypertension == 'Yes':
            hypertension = 1
        else:
            hypertension = 0

        if heart_disease == 'Yes':
            heart_disease = 1
        else:
            heart_disease = 0

        # Turn the inputs into a list
        list_of_inputs = [[age, bmi, glucose, hypertension, heart_disease]]
        # Using the list, feed it into the model so it can make a prediction
        print(list_of_inputs)
        pred = model.predict(list_of_inputs)
        # Grab the predicted probability for the prediction
        predicted_proba = model.predict_proba(list_of_inputs)
        predicted_proba = predicted_proba[0]

        # Change the output of pred to a string that will be printed out
        if pred[0] == 1:
            probability = predicted_proba[1] * 100
            probability = round(probability, 2)
            prediction = "The model predicts With a probability of {}% that you are AT RISK for a stroke".format(probability)
        else:
            probability = predicted_proba[1] * 100
            probability = round(probability, 2)
            prediction = "The model predicts With a probability of {}% that you are AT RISK for a stroke".format(probability)

        ## Return our inputs, list if inputs, and prediction message back to index.html
        return render_template('index.html',
                               returned_age=age,
                               returned_bmi=bmi,
                               returned_glucose=glucose,
                               returned_hyp=hypertension,
                               returned_heart=heart_disease,
                               returned_list=list_of_inputs,
                               returned_pred=prediction
                               # returned_prob=predicted_proba
                               )

    # return(flask.render_template('index.html'))


if __name__ == '__main__':
    app.run(debug=True)
