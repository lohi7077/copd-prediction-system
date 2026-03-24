from flask import Flask, render_template, request, session, flash
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import mysql.connector


app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

# Load models
scaler = joblib.load('scaler.pkl')
model = tf.keras.models.load_model('copd_cnn_model.h5')

# Feature names in correct order (after dropping columns)
feature_names = [
    'AGE', 'PackHistory', 'MWT1Best', 'FEV1', 'FEV1PRED', 'FVC',
    'FVCPRED', 'CAT', 'HAD', 'SGRQ', 'AGEquartiles', 'gender',
    'smoking', 'Diabetes', 'muscular', 'hypertension', 'AtrialFib', 'IHD'
]

# COPD severity mapping
severity_map = {
    0: "No COPD",
    1: "Mild COPD",
    2: "Moderate COPD",
    3: "Severe COPD",
    4: "Very Severe COPD"
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/AdminLogin")
def AdminLogin():
    return render_template('AdminLogin.html')


@app.route('/NewUser')
def NewUser():
    return render_template('NewUser.html')


@app.route('/UserLogin')
def UserLogin():
    return render_template('UserLogin.html')

@app.route('/Chat')
def Chat():
    return render_template('Chatbot.html')

@app.route("/adminlogin", methods=['GET', 'POST'])
def adminlogin():
    if request.method == 'POST':
        if request.form['uname'] == 'admin' and request.form['password'] == 'admin':
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='2medicalchatnewdb')
            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb")
            data = cur.fetchall()

            return render_template('AdminHome.html', data=data)
        else:
            flash("UserName or Password Incorrect!")

            return render_template('AdminLogin.html')

@app.route("/AdminHome")
def AdminHome():
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='2medicalchatnewdb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb")
    data = cur.fetchall()
    return render_template('AdminHome.html', data=data)


@app.route("/newuser", methods=['GET', 'POST'])
def newuser():
    if request.method == 'POST':
        name = request.form['uname']
        mobile = request.form['mobile']
        email = request.form['email']
        address = request.form['address']
        username = request.form['username']
        password = request.form['password']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='2medicalchatnewdb')
        cursor = conn.cursor()
        cursor.execute(
            "insert into regtb values('','" + name + "','" + mobile + "','" + email + "','"+ address +"','" + username + "','" + password + "')")
        conn.commit()
        conn.close()
        flash("Record Saved!")
    return render_template('UserLogin.html')


@app.route("/userlogin", methods=['GET', 'POST'])
def userlogin():
    if request.method == 'POST':
        username = request.form['uname']
        password = request.form['password']
        session['uname'] = request.form['uname']

        conn = mysql.connector.connect(user='root', password='', host='localhost', database='2medicalchatnewdb')
        cursor = conn.cursor()
        cursor.execute("SELECT * from regtb where username='" + username + "' and password='" + password + "'")
        data = cursor.fetchone()
        if data is None:

            flash("UserName Or Password Incorrect..!")
            return render_template('UserLogin.html', data=data)
        else:
            conn = mysql.connector.connect(user='root', password='', host='localhost', database='2medicalchatnewdb')
            cur = conn.cursor()
            cur.execute("SELECT * FROM regtb where username='" + username + "' and password='" + password + "'")
            data = cur.fetchall()

            flash("you are successfully logged in")
            return render_template('UserHome.html', data=data)


@app.route("/UserHome")
def UserHome():
    uname = session['uname']
    conn = mysql.connector.connect(user='root', password='', host='localhost', database='2medicalchatnewdb')
    cur = conn.cursor()
    cur.execute("SELECT * FROM regtb where username='" + uname + "' ")
    data = cur.fetchall()
    return render_template('UserHome.html', data=data)





@app.route('/Predict')
def Predict():
    return render_template('Predict.html')  # Render the HTML form
@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form

    # Create input array in correct order
    input_values = [
        float(form_data['age']),
        float(form_data['pack_history']),
        float(form_data['mwt1_best']),
        float(form_data['fev1']),
        float(form_data['fev1_pred']),
        float(form_data['fvc']),
        float(form_data['fvc_pred']),
        float(form_data['cat']),
        float(form_data['had']),
        float(form_data['sgrq']),
        float(form_data['age_quartiles']),
        int(form_data['gender']),
        int(form_data['smoking']),
        int(form_data['diabetes']),
        int(form_data['muscle']),
        int(form_data['hypertension']),
        int(form_data['atrial_fib']),
        int(form_data['ihd'])
    ]

    # Create DataFrame
    input_df = pd.DataFrame([input_values], columns=feature_names)

    # Scale features
    scaled_data = scaler.transform(input_df)

    # Reshape for CNN
    cnn_input = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1], 1)

    # Make prediction
    prediction = model.predict(cnn_input)
    pred_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    pre =''





    # Get severity description
    severity = severity_map.get(pred_class, "Unknown")
    print(severity)

    if severity == 'Mild COPD':
        pre = (
        'Mild COPD detected. Quit smoking if applicable. Begin light exercise (like walking), '
        'track symptoms, avoid lung irritants, and follow up regularly with a doctor.'
    )
    elif severity == 'Moderate COPD':
        pre = (
        'Moderate COPD. Important to quit smoking, use prescribed inhalers (bronchodilators), '
        'consider pulmonary rehabilitation, maintain a healthy diet, and monitor oxygen levels at home.'
    )

    elif severity == 'Severe COPD':
        pre = (
        'Severe COPD. Follow a strict treatment plan including long-acting inhalers and possibly corticosteroids. '
        'Oxygen therapy might be needed. Pulmonary rehabilitation is highly recommended. '
        'Avoid respiratory infections by staying vaccinated and practicing good hygiene.'
    )
    elif severity == 'Very Severe COPD':
        pre = (
        'Very Severe COPD. Requires intensive management. Oxygen therapy is often necessary. '
        'May need surgical options such as lung volume reduction or transplant evaluation. '
        'Avoid all pollutants, follow strict medication use, and consult with a pulmonologist regularly. '
        'Consider palliative care options for symptom relief and quality-of-life improvement.'
    )
    else:
        pre = 'Nil'

    # Prepare probabilities
    probabilities = {
        severity_map[i]: f"{float(p) * 100:.1f}%"
        for i, p in enumerate(prediction[0])
    }

    return render_template('result.html',
                           prediction=severity,
                           confidence=f"{confidence * 100:.1f}%",
                           probabilities=probabilities,pre=pre)



if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)

