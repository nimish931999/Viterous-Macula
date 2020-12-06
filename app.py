from flask import Flask, render_template,request,redirect
import numpy as np
#import tensorflow
from tensorflow.keras.models import load_model
import pandas as pd
app = Flask(__name__)
import os
print(os.listdir())

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/data', methods=['POST'])
def data():
    # i=1
    temp = request.form
    temp = pd.DataFrame(temp, index = [0])
    # data.to_csv('test.csv')
    data = pd.read_csv('test.csv')
    data = data.append(temp, ignore_index = True)
    # print(data)
    data.to_csv('test.csv')
    # data.to_csv('/test.csv')
    # data.headache.apply(lambda x: 1 if x =='yes' else 0)
    # data.near_reading_problems.apply(lambda x: 1 if x =='yes' else 0)
    # data.far_reading_problems.apply(lambda x: 1 if x =='yes' else 0)
    # data.watering_eyes.apply(lambda x: 1 if x =='yes' else 0)
    # data.dizziness.apply(lambda x: 1 if x =='yes' else 0)
    # data.eye_strain.apply(lambda x: 1 if x =='yes' else 0)
    # data.gender.apply(lambda x: 1 if x =='Female' else 0)

    df=pd.read_csv("test.csv")
    rd = data[['headache','near_reading_problems','far_reading_problems','watering_eyes','dizziness','eye_strain','age','gender','color_vision_r','spokes_r','long_range_vision_r','short_range_vision_r']]
    ld = data[['headache','near_reading_problems','far_reading_problems','watering_eyes','dizziness','eye_strain','age','gender','color_vision_l','spokes_l','long_range_vision_l','short_range_vision_l']]
    
    # columns_r = ['color_vision_r_b','color_vision_r_g','color_vision_r_r','headache','near_reading_problems','far_reading_problems','watering_eyes','dizziness','eye_strain','age','gender','spokes_r','long_range_vision_r','short_range_vision_r']
    # columns_l = ['color_vision_l_b','color_vision_l_g','color_vision_l_r','headache','near_reading_problems','far_reading_problems','watering_eyes','dizziness','eye_strain','age','gender','spokes_l','long_range_vision_l','short_range_vision_l']
    # # COLOR VISION 
    # # RIGHT
    # X_r = data_r.iloc[:, :].values
    # from sklearn.compose import ColumnTransformer
    # from sklearn.preprocessing import OneHotEncoder
    # ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [8])], remainder='passthrough')
    # X_r = np.array(ct.fit_transform(X_r))
    # RLD = pd.DataFrame(data = X_r,columns = columns_r)
    # RLD.to_csv('RLD.csv')
    
    # #LEFT
    # X_l = data_l.iloc[:, :].values
    # from sklearn.compose import ColumnTransformer
    # from sklearn.preprocessing import OneHotEncoder
    # ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [8])], remainder='passthrough')
    # X_l = np.array(ct.fit_transform(X_l))
    # FLD = pd.DataFrame(data = X_l,columns = columns_l)
    # FLD.to_csv('FLD.csv')
    
    
    
    
    # model_axis = load_model('model_axis_r.h5')
    # # AXIS RIGHT 
    # x_spokes_right = data.spokes_r.apply(lambda x: int(x))
    # axis_right = model_axis.predict(x_spokes_right)
    # axis_right = axis_right[-1]
    # axis_right = str(axis_right)
    # print(axis_right + "  Right Axis")

    # # AXIS LEFT 
    # x_spokes_left = data.spokes_l.apply(lambda x: int(x))
    # axis_left = model_axis.predict(x_spokes_left)
    # axis_left = axis_left[-1]
    # axis_left = str(axis_left)
    # print(axis_left + "  Left Axis")
    
    
    # model_addition = load_model('model_addition_r.h5')
    # # Addition Right
    # x_add_right = RLD[['color_vision_r_b','color_vision_r_g','color_vision_r_r','headache','near_reading_problems','far_reading_problems','watering_eyes','dizziness','eye_strain','age','gender','short_range_vision_r']]
    # add_right = model_addition.predict(x_add_right)
    # add_right = add_right[-1]
    # add_right = str(add_right)
    # print(add_right + "  Addition Right") 
    
    # # ADDITION LEFT
    # x_add_left = FLD[['color_vision_l_b','color_vision_l_g','color_vision_l_r','headache','near_reading_problems','far_reading_problems','watering_eyes','dizziness','eye_strain','age','gender','short_range_vision_l']]
    # add_left = model_addition.predict(x_add_left)
    # add_left = add_left[-1]
    # add_left = str(add_left)
    # print(add_left + " Addition Left")

    
    # rd=rd.drop(['color_vision_l','spokes_l','long_range_vision_l','short_range_vision_l'],axis=1)
    rd.to_excel(r'right_eye.xlsx')

    # ld=ld.drop(['color_vision_r','spokes_r','long_range_vision_r','short_range_vision_r'],axis=1)
    ld.to_excel(r'left_eye.xlsx')

    # #LEFT
    ld=ld.replace(to_replace=['No','Yes','no', 'yes','Female','Male','female','male'], value=[0,1,0,1,1,0,1,0])

    X = ld.iloc[:, :].values
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [8])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:, 9:10])
    X[:, 9:10] = imputer.transform(X[:, 9:10])
    columns_l = ['color_vision_l_b','color_vision_l_g','color_vision_l_r','headache','near_reading_problems','far_reading_problems','watering_eyes','dizziness','eye_strain','age','gender','spokes_l','long_range_vision_l','short_range_vision_l']
    FLD = pd.DataFrame(data = X,   columns = columns_l) 
    FLD = FLD.fillna(0)
    FLD.to_excel(r'FLD.xlsx')

    # #RIGHT

    rd=rd.replace(to_replace=['No','Yes','no', 'yes','Female','Male','female','male'], value=[0,1,0,1,1,0,1,0])

    X = rd.iloc[:, :].values
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [8])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:, 9:10])
    X[:, 9:10] = imputer.transform(X[:, 9:10])
    columns_r = ['color_vision_r_b','color_vision_r_g','color_vision_r_r','headache','near_reading_problems','far_reading_problems','watering_eyes','dizziness','eye_strain','age','gender','spokes_r','long_range_vision_r','short_range_vision_r']
    RLD = pd.DataFrame(data = X,   columns = columns_r) 
    RLD = RLD.fillna(0)
    RLD.to_excel(r'RLD.xlsx')

    ## IMPORTING MODELS
    from tensorflow import keras
    model_axis = keras.models.load_model('axis.h5')
    model_sph = keras.models.load_model('model_spherical_r.h5')
    model_cyl = keras.models.load_model('model_cyl_r.h5')
    model_add = keras.models.load_model('model_addition_r.h5')

    ## AXIS
    ## RIGHT
    data=pd.read_excel("RLD.xlsx",index=False)
    # data.drop("Unnamed: 0",axis=1)
    x = data.spokes_r
    pred = model_axis.predict(x)
    r_axis = pred[-1]
    r_axis=str(r_axis)

    ## LEFT
    data=pd.read_excel("FLD.xlsx",index=False)
    # data.drop("Unnamed: 0",axis=1)
    x = data.spokes_l
    pred = model_axis.predict(x)
    l_axis = pred[-1]
    l_axis=str(l_axis)


    ###SPHERICAL PREDICTION
    ### RIGHT
    data=pd.read_excel("RLD.xlsx",index=False)
    x = data[['color_vision_r_b','color_vision_r_g','color_vision_r_r','headache','near_reading_problems','far_reading_problems','watering_eyes','dizziness','long_range_vision_r']]
    pred = model_sph.predict(x)
    r_sph = pred[-1]
    r_sph=str(r_sph)

    ### LEFT
    data=pd.read_excel("FLD.xlsx",index=False)
    x = data[['color_vision_l_b','color_vision_l_g','color_vision_l_r','headache','near_reading_problems','far_reading_problems','watering_eyes','dizziness','long_range_vision_l']]
    pred = model_sph.predict(x)
    l_sph = pred[-1]
    l_sph=str(l_sph)


    ###CYLENDRICAL PREDICTION
    ### RIGHT
    data=pd.read_excel("RLD.xlsx",index=False)
    x = data[['color_vision_r_b','color_vision_r_g','color_vision_r_r','headache','near_reading_problems','far_reading_problems','watering_eyes','dizziness','long_range_vision_r']]
    pred = model_cyl.predict(x)
    r_cyl = pred[-1]
    r_cyl=str(r_cyl)

    ### LEFT
    data=pd.read_excel("FLD.xlsx",index=False)
    x = data[['color_vision_l_b','color_vision_l_g','color_vision_l_r','headache','near_reading_problems','far_reading_problems','watering_eyes','dizziness','long_range_vision_l']]
    pred = model_cyl.predict(x)
    l_cyl = pred[-1]
    l_cyl=str(l_cyl)


    ###ADDITION PREDICTION
    ## RIGHT
    data=pd.read_excel("RLD.xlsx",index=False)
    x = data[['color_vision_r_b','color_vision_r_g','color_vision_r_r','headache','near_reading_problems','far_reading_problems','watering_eyes','dizziness','eye_strain','age','gender','short_range_vision_r']]
    pred = model_add.predict(x)
    r_add = pred[-1]
    r_add=str(r_add)

    ### LEFT
    data=pd.read_excel("FLD.xlsx",index=False)
    x = data[['color_vision_l_b','color_vision_l_g','color_vision_l_r','headache','near_reading_problems','far_reading_problems','watering_eyes','dizziness','eye_strain','age','gender','short_range_vision_l']]
    pred = model_add.predict(x)
    l_add = pred[-1]
    l_add=str(l_add)

    ### PRINT
    print("====================================================Axis====================================================")
    print('Right Eye Axis: ' + r_axis)
    print("Left Eye Axis: " + l_axis)
    print("==================================================Spherical=================================================")
    print("Right Eye Spherical: " + r_sph)
    print("Left Eye Spherical: " + l_sph)
    print("=================================================Cylendrical================================================")
    print("Right Eye Cylendrical: " + r_cyl)
    print("Left Eye Cylendrical: " + l_cyl)
    print("===================================================Addition=================================================")
    print("Right Eye Addition: " + r_add)
    print("Left Eye Addition: " + l_add)

    # key = r_axis
    # value = r_axis

    res = { 'Axis' : [ r_axis , l_axis],
            'Spherical' : [ r_sph , l_sph],
            'Cylindrecal' : [ r_cyl , l_cyl],
            'Addition' : [ r_add , l_add]
            }


    # return redirect('/')
    return render_template("test.html",result = res)
if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8000, debug=True)


 