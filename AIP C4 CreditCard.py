from flask import Flask, render_template, flash, request, url_for, redirect, session
# from flask_admin import Admin
from Models._user import User, db, connect_to_db
from Forms.forms import RegistrationForm, LoginForm
from passlib.hash import sha256_crypt
import os.path
import csv
import gc, os
from functools import wraps
import pickle
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd

ALLOWED_EXTENSIONS = set(['csv','xls'])


app = Flask(__name__)
conn = 'sqlite:///'+ os.path.abspath(os.getcwd())+"/DataBases/test.db"
xgb_model = pickle.load(open('Models/xgb_model.pkl','rb'))

connect_to_db(app,conn)


# EMAIL_ADDRESS = "financialadvisor20113@gmail.com"
# EMAIL_PASSWORD = "finaadvi"
app.config['SECRET_KEY'] = os.urandom(24)

@app.route('/', methods=['GET','POST'])
def main():
    return render_template('main.html')

@app.route('/about', methods=['GET','POST'])
def about():
    return render_template('about.html') 



@app.route('/register/', methods=['GET','POST'])
def register_page():
    try:
        form = RegistrationForm(request.form)
        if request.method == 'POST':
            _username = request.form['username']
            _email = request.form['email']
            _password = sha256_crypt.encrypt(str(form.password.data))
            user = User(username = _username, email = _email, password = _password)
            db.create_all()
            if User.query.filter_by(username=_username).first() is not None:
                flash('User Already registered with username {}'.format(User.query.filter_by(username=_username).first().username), "warning")
                return render_template('register.html', form=form)
            if User.query.filter_by(email=_email).first() is not None:
                flash('Email is already registered with us {}'.format(User.query.filter_by(email=_email).first().username), "warning")
                return render_template('register.html', form=form)
            flash("Thank you for registering!", "success")
            db.session.add(user)
            db.session.commit()
            db.session.close()
            gc.collect()
            session['logged_in'] = True
            session['username'] = _username
            session.modified = True
            return redirect(url_for('dashboard'))
        return render_template('register.html', form=form)
    except Exception as e:
        return render_template('error.html',e=e)

def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args,**kwargs)
        else:
            flash('You need to login first!', "warning")
            return redirect(url_for('login_page'))
    return wrap

def already_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            flash("You are already logged in!", "success")
            return redirect(url_for('dashboard'))
        else:
            return f(*args, **kwargs)
    return wrap

def verify(_username, _password):
    if User.query.filter_by(username=_username).first() is None:
        flash("No such user found with this username", "warning")
        return False
    if not sha256_crypt.verify(_password, User.query.filter_by(username=_username).first().password):
        flash("Invalid Credentials, password isn't correct!", "danger")
        return False
    return True


@app.route('/login/', methods=['GET','POST'])
# @already_logged_in
def login_page():
    try:

        form = LoginForm(request.form)
        if request.method == 'POST':

            _username = request.form['username']
            _password = request.form['password']


            if verify(_username, _password) is False:
                return render_template('login.html', form=form)
            session['logged_in'] = True
            session['username'] = _username
            gc.collect()
            return redirect(url_for('dashboard'))

        return render_template('login.html', form=form)


    except Exception as e:
        return render_template('error.html',e=e)


@app.route('/logout/')
@login_required
def logout():
    session.clear()
    gc.collect()
    flash("You have been logged out!", "success")
    return redirect(url_for('login_page'))
 
import csv

required_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

app.secret_key = "your_secret_key"

@app.route('/uploadfile', methods=['GET', 'POST'])
def uploadfile():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            if file.filename.endswith('.csv'):
                filename = file.filename
                save_location = os.path.join('CSV_Folder', filename)
                file.save(save_location)
                session['filename'] = filename
                
                # Check if the file has the required columns
                try:
                    with open(save_location, 'r') as f:
                        reader = csv.DictReader(f)
                        columns = reader.fieldnames
                        if not all(column in columns for column in required_columns):
                            os.remove(save_location)  # Remove the file if it doesn't have the required columns
                            flash("The uploaded file doesn't have the required columns", "warning")
                            return render_template('uploadfile.html')
                except Exception as e:
                    os.remove(save_location)  # Remove the file if there's an exception
                    flash("The uploaded file doesn't have the required columns Format should be: " + str(required_columns), "warning")
                    return render_template('uploadfile.html')
                
                return redirect(url_for('dashboard'))
            else:
                flash("Invalid file type. Please upload a .csv file", "warning")
                return render_template('uploadfile.html')
        
    return render_template('uploadfile.html')


# @app.route('/uploadfile', methods=['GET', 'POST'])
# def uploadfile():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file and file.filename:
#             if file.filename.endswith('.csv'):
#                 filename = file.filename
#                 save_location = os.path.abspath(os.path.join('CSV_Folder', filename))
#                 session['filename'] = filename
                
#                 # Check if the file has the required columns
#                 try:
#                     with open(save_location, 'r') as f:
#                         reader = csv.DictReader(f)
#                         columns = reader.fieldnames
#                         if not all(column in columns for column in required_columns):
#                             flash("The uploaded file doesn't have the required columns", "warning")
#                             return render_template('uploadfile.html')
#                 except Exception as e:
#                     flash("The uploaded file doesn't have the required columns Format should be: " + str(required_columns) , "warning")
                    
#                     return render_template('uploadfile.html')
                
#                 return redirect(url_for('dashboard'))
#             else:
#                 flash("Invalid file type. Please upload a .csv file", "warning")
#                 return render_template('uploadfile.html')
        
#     return render_template('uploadfile.html')



@app.route('/manualdata', methods=['GET', 'POST'])
def manualdata():
    if request.method == 'POST':
        return render_template('manuals.html')
    return render_template('manuals.html')

@app.route('/manuals', methods=['GET', 'POST'])
def manuals():
    if request.method == 'POST': 
        textarea_value = request.form['row_value']
        textarea_value = textarea_value.replace(", ", ",")
        textarea_value = textarea_value.split(',')
        print(textarea_value)
        input_data = pd.DataFrame({
            "Time": [float(textarea_value[0])],
            "V1": [float(textarea_value[1])],
            "V2": [float(textarea_value[2])],
            "V3": [float(textarea_value[3])],
            "V4": [float(textarea_value[4])],
            "V5": [float(textarea_value[5])],
            "V6": [float(textarea_value[6])],
            "V7": [float(textarea_value[7])],
            "V8": [float(textarea_value[8])],
            "V9": [float(textarea_value[9])],
            "V10": [float(textarea_value[10])],
            "V11": [float(textarea_value[11])],
            "V12": [float(textarea_value[12])],
            "V13": [float(textarea_value[13])],
            "V14": [float(textarea_value[14])],
            "V15": [float(textarea_value[15])],
            "V16": [float(textarea_value[16])],
            "V17": [float(textarea_value[17])],
            "V18": [float(textarea_value[18])],
            "V19": [float(textarea_value[19])],
            "V20": [float(textarea_value[20])],
            "V21": [float(textarea_value[21])],
            "V22": [float(textarea_value[22])],
            "V23": [float(textarea_value[23])],
            "V24": [float(textarea_value[24])],
            "V25": [float(textarea_value[25])],
            "V26": [float(textarea_value[26])],
            "V27": [float(textarea_value[27])],
            "V28": [float(textarea_value[28])],
            "Amount": [float(textarea_value[29])]
            })
        print(input_data)
        
        scaler = RobustScaler().fit(input_data[["Time", "Amount"]])        
        input_data[["Time", "Amount"]] = scaler.transform(input_data[["Time", "Amount"]])

        predictions = xgb_model.predict(input_data)
        
    
        is_fraud = []
        for predictions in predictions:
            if predictions == 0:
                is_fraud.append(False)  # non-fraud
            else:
                is_fraud.append(True)  # fraud
        context = {
            'input_data': input_data,
            'predictions': predictions,
            'enumerate': enumerate,
            'is_fraud': is_fraud,
            'Time': "Time",
            'Amount': float(textarea_value[29]),
            }
        return render_template('results_single.html', **context)

    return render_template('manuals.html')


@app.route('/results', methods=['GET', 'POST'])
def results():
    return render_template('results.html')


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    folder_path = os.path.join(app.root_path, 'CSV_Folder')
    files = os.listdir(folder_path)
    context = {
        'files': files,
        'enumerate': enumerate
    }
    return render_template('dashboard.html', **context)  
    

@app.route('/filedata/<filename>', methods=['GET', 'POST'])
def filedata(filename):
    if not filename:
        flash('Please upload a file first!')
        return redirect(url_for('upload'))

    filepath = os.path.join('CSV_Folder',filename)
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        column = [column for column in reader]
        row = [row for row in reader]
        context= {
            'column':column, 
            'row':row,
            'filename':filename,
            'enumerate': enumerate
        }
    return render_template('filedata.html', **context)


@app.route('/pred_file/<filename>', methods=['GET', 'POST'])
def pred_file(filename):
    if not filename:
        flash('Please upload a file first!')
        return redirect(url_for('upload'))
    
    filepath = os.path.join('CSV_Folder',filename)
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        column = [column for column in reader]
        row = [row for row in reader]
        print(column, row)
    df = pd.read_csv(filepath)
    df.isnull().sum()
    df.isnull().shape[0]
    
    scaler = RobustScaler().fit(df[["Time", "Amount"]])
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

    predictions = xgb_model.predict(df)
    is_fraud = []
    for pred in predictions:
        if pred == 0:
            is_fraud.append(False)  # non-fraud
        else:
            is_fraud.append(True)  # fraud
    
    context= {
        'column':column, 
        'row':row,
        'filename':filename,
        'enumerate': enumerate,
        'is_fraud': is_fraud
    }
    return render_template('pred_filedata.html', **context)



from flask import send_file

@app.route('/download-csv/<filename>')
def download_csv(filename):
    # Set the file path
    file_path = os.path.join('CSV_Folder', filename)

    # Send the file for download
    return send_file(file_path, as_attachment=True)


# @app.route('/analysis/<filename>', methods=['GET', 'POST'])
# def analysis(filename):
#     if not filename:
#         flash('Please upload a file first!')
#         return redirect(url_for('upload'))
    
#     filepath = os.path.join('CSV_Folder',filename)
#     with open(filepath, newline='') as csvfile:
#         reader = csv.DictReader(csvfile)
#         column = [column for column in reader]
#         row = [row for row in reader]
#     df = pd.read_csv(filepath)
#     df.isnull().sum()
#     df.isnull().shape[0]
    
#     scaler = RobustScaler().fit(df[["Time", "Amount"]])
#     df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

#     predictions = xgb_model.predict(df)
#     is_fraud = [prediction == 1 for prediction in predictions]
    
    
#     explainer = lime.lime_tabular.LimeTabularExplainer(df.values, feature_names=df.columns, class_names=['Non-Fraud', 'Fraud'], discretize_continuous=False)

#     # generate LIME explanation for a random sample of the predicted data
#     sample_size = min(5, len(df))
#     explainer_samples = np.random.randint(len(df), size=sample_size)
#     for sample_idx in explainer_samples:
#         sample = df.iloc[sample_idx]
#         exp = explainer.explain_instance(sample.values, xgb_model.predict_proba, num_features=len(df.columns))
#         print(exp.as_list())
    
#     df['is_fraud'] = is_fraud
#     df['prediction'] = predictions
#     df.loc[~df['is_fraud'], 'prediction'] = 0
    
#     y = df['is_fraud']
#     labels = ["Fraud", "Non-Fraud"]
#     values = y.value_counts().tolist()  

#     context= {
#         'column':column, 
#         'row':row,
#         'filename':filename,
#         'enumerate': enumerate,
#         'is_fraud': is_fraud,
#         'values' : values,
#         'labels' : labels
#     }
#     return render_template('analysis.html', **context)


import lime
import lime.lime_tabular
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

@app.route('/analysis/<filename>', methods=['GET', 'POST'])
def analysis(filename):
    if not filename:
        flash('Please upload a file first!')
        return redirect(url_for('upload'))
    
    filepath = os.path.join('CSV_Folder',filename)
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        column = [column for column in reader]
        row = [row for row in reader]
    df = pd.read_csv(filepath)
    df.isnull().sum()
    df.isnull().shape[0]
    
    scaler = RobustScaler().fit(df[["Time", "Amount"]])
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

    predictions = xgb_model.predict(df)
    is_fraud = [prediction == 1 for prediction in predictions]
    
    explainer = lime.lime_tabular.LimeTabularExplainer(df.values, feature_names=df.columns, class_names=['Non-Fraud', 'Fraud'], discretize_continuous=False)
    sample_size = min(5, len(df))
    explanations = {}
    for i in range(sample_size):
        exp = explainer.explain_instance(df.values[i], xgb_model.predict_proba, num_features=len(df.columns))
        explanations[i] = exp.as_list()
    
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(df)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.show()
    
    df['is_fraud'] = is_fraud
    df['prediction'] = predictions
    df.loc[~df['is_fraud'], 'prediction'] = 0
    
    y = df['is_fraud']
    labels = ["Fraud", "Non-Fraud"]
    values = y.value_counts().tolist()  

    context= {
        'column':column, 
        'row':row,
        'filename':filename,
        'enumerate': enumerate,
        'is_fraud': is_fraud,
        'values' : values,
        'labels' : labels,
        'explanations': explanations,
        'tsne_results': tsne_results
    }
    return render_template('analysis.html', **context)

if __name__ == '__main__':
    app.run()
