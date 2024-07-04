import numpy as np
import pandas as pd

import xgboost as xgb
from xgboost import XGBClassifier 

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import joblib

# M = 0, F = 1
# Rainy = 0, Sunny = 1, Cloudy = 2 , Snowy = 3


def data_generation():
    sample_id1 = range(0,20000)
    age1 = np.random.randint(0,60,size= 20000)
    weather1 = np.random.randint(0,5,size= 20000)
    sex1 = [0]*20000
    temp1 = np.random.randint(-15,40, size = 20000)
    food1 = [0]*8000 + [1]*8000 + [2]*4000
    data1 = pd.DataFrame({
        'sample_id' : sample_id1, 
        'age' : age1,
        'weather' : weather1,
        'sex' : sex1,
        'temperature' : temp1,
        'food' : food1
        })

    sample_id2 = range(20000,40000)
    age2 = np.random.randint(0,60,size= 20000)
    weather2 = np.random.randint(0,5,size= 20000)
    sex2 = [1]*20000
    temp2 = np.random.randint(-15,40,size = 20000)
    food2 = [3]*8000 + [4]*8000 + [5]*4000
    data2 = pd.DataFrame({
        'sample_id' : sample_id2, 
        'age' : age2,
        'weather' : weather2,
        'sex' : sex2,
        'temperature' : temp2,
        'food' : food2
        })

    sample_id3 = range(40000,60000)
    age3 = np.random.randint(0,60,size= 20000)
    weather3 = [1]*20000
    sex3 = np.random.randint(0,2,size= 20000)
    temp3 = np.random.randint(30,40,size = 20000)
    food3 = [6]*20000
    data3 = pd.DataFrame({
        'sample_id' : sample_id3, 
        'age' : age3,
        'weather' : weather3,
        'sex' : sex3,
        'temperature' : temp3,
        'food' : food3
        })

    sample_id4 = range(60000,80000)
    age4 = np.random.randint(0,60,size= 20000)
    weather4 = [1]*10000 + [3]*10000
    sex4 = np.random.randint(0,2,size= 20000)
    temp4 = np.random.randint(-15,25,size = 20000)
    food4 = [7]*7000 + [12]*13000
    data4 = pd.DataFrame({
        'sample_id' : sample_id4, 
        'age' : age4,
        'weather' : weather4,
        'sex' : sex4,
        'temperature' : temp4,
        'food' : food4
        })

    sample_id5 = range(80000,100000)
    age5 = np.random.randint(0,60,size= 20000)
    weather5 = [0]*20000
    sex5 = np.random.randint(0,2,size= 20000)
    temp5 = np.random.randint(15,30,size = 20000)
    food5 = [7]*7000 + [12]*13000
    data5 = pd.DataFrame({
        'sample_id' : sample_id5, 
        'age' : age5,
        'weather' : weather5,
        'sex' : sex5,
        'temperature' : temp5,
        'food' : food5
        })

    sample_id6 = range(100000,120000)
    age6 = np.random.randint(0,60,size= 20000)
    weather6 = np.random.randint(0,4,size= 20000)
    sex6 = np.random.randint(0,2,size= 20000)
    temp6 = np.random.randint(-15,45,size = 20000)
    food6 = np.random.randint(0,15,size=20000)
    data6 = pd.DataFrame({
        'sample_id' : sample_id6, 
        'age' : age6,
        'weather' : weather6,
        'sex' : sex6,
        'temperature' : temp6,
        'food' : food6
        })
    
    sample_id7 = range(120000,150000)
    age7 = np.random.randint(0,60,size=30000)
    weather7 = np.random.randint(0,4,size= 30000)
    sex7 = np.random.randint(0,2,size= 30000)
    temp7 = np.random.randint(-15,45,size = 30000)
    food7 = np.repeat(np.array([10,12]),15000)
    data7 = pd.DataFrame({
        'sample_id' : sample_id7, 
        'age' : age7,
        'weather' : weather7,
        'sex' : sex7,
        'temperature' : temp7,
        'food' : food7
        })
    
    sample_id8 = range(150000,175000)
    age8 = np.random.randint(0,60,size=25000)
    weather8 = np.random.randint(0,4,size= 25000)
    sex8 = np.random.randint(0,2,size= 25000)
    temp8 = np.random.randint(30,45,size = 25000)
    food8 = np.random.randint(6,7,size=25000)
    data8 = pd.DataFrame({
        'sample_id' : sample_id8, 
        'age' : age8,
        'weather' : weather8,
        'sex' : sex8,
        'temperature' : temp8,
        'food' : food8
        })

    sample_id9 = range(175000,200000)
    age9 = np.random.randint(0,60,size=25000)
    weather9 = np.random.randint(0,4,size= 25000)
    sex9 = np.random.randint(0,2,size= 25000)
    temp9 = np.random.randint(10,25,size = 25000)
    food9 = np.random.randint(7,8,size=25000)
    data9 = pd.DataFrame({
        'sample_id' : sample_id9, 
        'age' : age9,
        'weather' : weather9,
        'sex' : sex9,
        'temperature' : temp9,
        'food' : food9
        })
    
    sample_id10 = range(200000,230000)
    age10 = np.random.randint(0,60,size= 30000)
    weather10 = np.random.randint(0,4,size= 30000)
    sex10 = [0]*30000
    temp10 = np.random.randint(-15,40, size = 30000)
    food10 = [0]*10000 + [1]*10000 + [2]*10000
    data10 = pd.DataFrame({
        'sample_id' : sample_id10, 
        'age' : age10,
        'weather' : weather10,
        'sex' : sex10,
        'temperature' : temp10,
        'food' : food10
        })
    
    sample_id11 = range(230000,260000)
    age11 = np.random.randint(0,60,size= 30000)
    weather11 = np.random.randint(0,4,size= 30000)
    sex11 = [1]*30000
    temp11 = np.random.randint(-15,40,size = 30000)
    food11 = [3]*10000 + [4]*10000 + [5]*10000
    data11 = pd.DataFrame({
        'sample_id' : sample_id11, 
        'age' : age11,
        'weather' : weather11,
        'sex' : sex11,
        'temperature' : temp11,
        'food' : food11
        })
    
    sample_id12 = range(260000,300000)
    age12 = np.random.randint(0,60,size= 40000)
    weather12 = np.random.randint(0,4,size= 40000)
    sex12 = np.random.randint(0,2,size=40000)
    temp12 = np.random.randint(-15,40,size = 40000)
    food12 = [9,13,14,11,8]*8000
    data12 = pd.DataFrame({
        'sample_id' : sample_id12, 
        'age' : age12,
        'weather' : weather12,
        'sex' : sex12,
        'temperature' : temp12,
        'food' : food12
        })
    
    data = pd.concat([data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12])
    print(f"Data_size : {data.shape}")
    print(f"Data_type : {type(data)}")
    return data

def skf(data):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    
    split_data = np.array(data.iloc[:,1:5])
    label = np.array(data.iloc[:,-1])
    
    j=0
    for train_index, test_index in skf.split(split_data,label):
        x_train, x_test = split_data[train_index,:], split_data[test_index,:]
        y_train, y_test = label[train_index], label[test_index]
        j=j+1
        if j==1: ### CV1 test
            break
    return x_train,x_test,y_train,y_test

def testdata(input_data):
    #categorizing sex
    if input_data[2] == 'Male':
        input_data[2] = 0
    else:
        input_data[2] = 1

    #categorizing weather
    if input_data[1] == 'Rainy':
        input_data[1] = 0
    elif input_data[1] == 'Sunny':
        input_data[1] = 1
    elif input_data[1] == 'Cloudy':
        input_data[1] = 2
    elif input_data[1] == 'Snowy':
        input_data[1] = 3
        
    t_data = np.array(input_data)
    test_data = t_data.reshape(1,len(t_data))

    return test_data

def model_train():
    data = data_generation()
    x_train, x_test, y_train, y_test = skf(data)
    print(f'Train_data : X_train = {x_train.shape}, y_train = {len(y_train)}')
    print(f'Test_data : X_test = {x_test.shape}, y_test = {len(y_test)}')
    
    std = StandardScaler()
    std.fit(x_train)
    scaled_x_train = std.transform(x_train)
    scaled_x_test = std.transform(x_test)
    
    dtrain = xgb.DMatrix(data=scaled_x_train, label=y_train, feature_names = ['age','weather','sex''temperature'])
    dval = xgb.DMatrix(data=scaled_x_test, label=y_test, feature_names = ['age','weather','sex','temperature'])
    
    #dtrain = xgb.DMatrix(data=x_train, label=y_train)
    #dval = xgb.DMatrix(data=x_test, label=y_test)

    wlist = [(dtrain, 'train'),(dval, 'eval')]
    params = {
        'max_depth':4, 
        'eta':0.05,  
        'objective':'multi:softprob', 
        'eval_metric':'mlogloss', 
        'num_class' : 15
    }

    model = xgb.train(params = params, dtrain=dtrain, num_boost_round = 500, early_stopping_rounds=50, evals=[(dtrain,'train'),(dval,'eval')])
    return model

def acc():
    data = data_generation()
    x_train, x_test, y_train, y_test = skf(data)
     
    model = XGBClassifier(n_estimators=5000, learning_rate=0.001, max_depth=6, random_state = 0,tree_method='gpu_hist', gpu_id=0)
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    
    return accuracy

def recommend(personal):
    model = joblib.load('./Personal_Recommendation/saved_model.pkl')
    
    input_data = testdata(personal)
    dtest = xgb.DMatrix(data=input_data)
    y_pred = model.predict(dtest)
    y_pred = y_pred[0] 
    yy = y_pred.argsort()
    yyy = yy[::-1]
    final = yyy[0:5]
    final = final.tolist()
    food_list = ['제육', '돈까스', '국밥', '떡볶이', '파스타', '마라탕', '냉면', '칼국수', '햄버거', '피자', '김밥', '짜장면', '라면', '아구찜', '핫도그']
    reco = []
    for i in final:
        f = food_list[i]
        reco.append(f)
    print(f'슬라이드쇼 후보군 : {reco[0]},{reco[1]},{reco[2]},{reco[3]},{reco[4]}')
    return final

##model_training
#trainned_model = model_train()
#acc()

##recommend
#personal1 = [29,'Rainy','Male',38]
#personal2 = [35,'Sunny','Female',23]
#personal3 = [45,'Snowy','Male',0]
#recommended_menu_id = recommend(personal3)





