from flask import Flask, render_template, request, redirect, flash
import datetime
import ml

app = Flask(__name__)
app.config['SECRET_KEY'] = 'YARR'


@app.route("/", methods=['POST', 'GET'])
def predict_Aqi():

    if request.method == 'POST':
        try:
            date_string = request.form['daate']
            model = request.form['moodel']
            date = datetime.datetime.strptime(date_string, "%Y-%m-%d")
            july12 = datetime.datetime.strptime('2023-07-12', "%Y-%m-%d")
            present = datetime.datetime.strptime('2023-07-07', "%Y-%m-%d")
            if date < present:
                flash('We do not have data prior to the date of 7th July!', 'Warning')
                return render_template('homepage.html')
            if date >= july12:
                flash('We do not have data beyond the date of 11th July!', 'Warning')
                return render_template('homepage.html')
            else:
                prediction = ml.make_prediction(date_string, model)

                if prediction == 0:
                    flash("The air quality on this day will likely be Good", 'good')
                elif prediction == 1:
                    flash("The air quality on this day will likely be Moderate", 'moderate')
                elif prediction == 2:
                    flash("The air quality on this day will likely be Unhealthy for sensitive groups", 'unhealthyfsg')
                elif prediction == 3:
                    flash("The air quality on this day will likely be Unhealthy", 'unhealthy')
                elif prediction == 4:
                    flash("The air quality on this day will likely be Very Unhealthy", 'vunhealthy')
                elif prediction == 5:
                    flash("The air quality on this day will likely be Hazardous", 'hazardous')

                return render_template('homepage.html')

        except Exception as errrr:

            return render_template('error.html', err=errrr)

    return render_template('homepage.html')


@app.route("/err", methods=['POST', 'GET'])
def err():
    return render_template('error.html')


if __name__ == '__main__':
    Flask.run(app)
