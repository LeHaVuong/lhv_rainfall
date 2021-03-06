import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse
import seaborn as sns
import statsmodels.api as sm
# import itertools
from statsmodels.tsa.arima_model import ARIMA, ARMA
import warnings
warnings.filterwarnings("ignore")
from datetime import date, datetime
import streamlit as st
from sklearn.metrics import mean_squared_error
import math

data = pd.read_csv('data_rain.csv',skiprows=10)

data['DATE'] = data['YEAR'].astype(str) + '-' + data['MO'].astype(str) + '-' + data['DY'].astype(str)

df = data[['DATE','PRECTOT']]

df = df.drop(df[df.PRECTOT <0].index)

df_ts = pd.DataFrame()
df_ts['ds'] = pd.to_datetime(df['DATE'])
df_ts['y'] = df['PRECTOT']

# Data theo ngay
df_ts1 = df_ts.copy(deep=False)

df_ts1.index = pd.to_datetime(df_ts1.ds)
df_ts1 = df_ts1.drop(['ds'],axis=1)

# Data theo thang
df_ts2 = df_ts1.resample('MS').mean()


# Build Model theo ngay
train_data1, test_data1 = df_ts1[0:int(len(df_ts1)*0.8)], df_ts1[int(len(df_ts1)*0.8):]
train_ar1 = train_data1['y'].values
test_ar1 = test_data1['y'].values

history1 = [x for x in train_ar1]
predictions1 = list()
for t in range(len(test_ar1)):
    model1 = ARIMA(history1, order=(2,1,0)) 
    model_fit1 = model1.fit(disp=0)
    output1 = model_fit1.forecast()
    yhat1 = output1[0]
    predictions1.append(yhat1)
    obs1 = test_ar1[t]
    history1.append(obs1)
    # print('predicted=%f, expected=%f' % (yhat, obs))
error1 = mean_squared_error(test_ar1, predictions1)
rmse1 = np.sqrt(np.mean(predictions1-test_ar1)**2) 

# Build Model theo thang
train_data2, test_data2 = df_ts2[0:int(len(df_ts2)*0.8)], df_ts2[int(len(df_ts2)*0.8):]
train_ar2 = train_data2['y'].values
test_ar2 = test_data2['y'].values

history2 = [x for x in train_ar2]
predictions2 = list()
for t in range(len(test_ar2)):
    model2 = ARIMA(history2, order=(2,1,0)) 
    model_fit2 = model2.fit(disp=0)
    output2 = model_fit2.forecast()
    yhat2 = output2[0]
    predictions2.append(yhat2)
    obs2 = test_ar2[t]
    history2.append(obs2)
    # print('predicted=%f, expected=%f' % (yhat, obs))
error2 = mean_squared_error(test_ar2, predictions2)
rmse2 = np.sqrt(np.mean(predictions2-test_ar2)**2)

# Part 2: Show project's result with Streamlit

st.title("Data Science")
st.header('Rainfall Prediction Project')
st.subheader('Make new Prediction')

menu = ['Overview','Build Project','New Prediction By Day','New Prediction By Month']
choice = st.sidebar.selectbox('Menu',menu)
if choice == 'Overview':
    st.subheader('Overview')
    st.write("""
    Th??nh ph??? H??? Ch?? Minh n???m trong v??ng nhi???t ?????i gi?? m??a c???n x??ch ?????o. C??ng nh?? c??c t???nh ??? Nam b???, ?????c ??i???m chung c???a kh?? h???u-th???i ti???t TPHCM l?? nhi???t ????? cao ?????u trong n??m v?? c?? hai m??a m??a - kh?? r?? r??ng l??m t??c ?????ng chi ph???i m??i tr?????ng c???nh quan s??u s???c. M??a m??a t??? th??ng 5 ?????n th??ng 11, m??a kh?? t??? th??ng 12 ?????n th??ng 4 n??m sau. X??t y???u t??? l?????ng m??a, c?? h??n 90% l?????ng m??a t???p trung v??o th??ng 5 ?????n th??ng 11. Hi???n nay ???? c?? s??? bi???n thi??n v??? l?????ng m??a, t???o ra ng???p c???c b???. C??c d??? b??o cho th???y r???ng t???ng l?????ng m??a h??ng n??m c?? xu h?????ng t??ng d???n tuy kh??ng nhi???u nh??ng s??? c?? s??? bi???n thi??n gi???a c??c m??a l???n h??n. Ng???p c???c b??? v?? v???y s??? t??ng v?? m??a l???n k???t h???p v???i b??o s??? tr??? n??n ph??? bi???n h??n. ??? ????y ta ph???i hi???u r???ng m??a nhi???u kh??ng ph???i l?? nguy??n nh??n ch??nh g??y ra ng???p ??? TPHCM v?? l??u v???c TPHCM ???? ???????c qu???n l?? ch???c ch???, nh??ng ng???p c???c b??? t??? m???t tr???n m??a c?????ng ????? cao s??? l?? m???t m???i ??e d???a l???n.

    Ng???p th?????ng xuy??n c?? th??? g??y s??? ph??t t??n c??c ch???t ?? nhi???m nghi??m tr???ng v?? g??y ra m???i ??e d???a v???i s???c kh???e c???ng ?????ng, s???n xu???t kinh t???, v?? c??c h??? sinh th??i. B??n c???nh ???? m??a nhi???u c??ng ???nh tr???c ti???p ?????n m???t s??? ng??nh ngh??? nh???t ?????nh nh?? x??y d???ng, du l???ch, v???n t???i???

    Ch??nh v?? v???y v???i m???c ????ch cung c???p v?? ph??? bi???n th??ng tin v??? l?????ng m??a ?????n t???t c??? m???i ng?????i, gi??p m???i ng?????i c?? c??i nh??n kh??ch quan v??? hi???n tr???ng m??a ??? th??nh ph???, t??c gi??? s??? x??y d???ng c??c m?? h??nh d??? ??o??n l?????ng m??a theo th???i gian trong t????ng lai d???a tr??n d??? li???u ???? thu th???p ???????c trong qu?? kh???.
    """)
elif choice == 'Build Project':
    st.subheader('Build Project')
    st.write('### Data preprocessing')
    st.write('### Show data:')
    st.table(df_ts.head(5))

    st.write('### Build model and evaluation with data by date:')
    st.write('Root mean square error (date): {}'.format (round(rmse1,2)))
    
    fig1,ax1 = plt.subplots()
    # ax1.figure(figsize=(12,7))
    ax1.plot(df_ts1['y'], 'green', color='blue', label='Training Data')
    ax1.plot(test_data1.index, predictions1, color='green', marker='o', linestyle='dashed', 
            label='Predicted')
    ax1.plot(test_data1.index, test_data1['y'], color='red', label='Actual')
    ax1.set_title('Rainfall Prediction')
    ax1.set_xlabel('Dates')
    ax1.set_ylabel('mm/day')
    st.pyplot(fig1)

    st.write('### Build model and evaluation with data by month:')
    st.write('Root mean square error (month): {}'.format (round(rmse2,2)))
    
    fig2,ax2 = plt.subplots()
    # ax2.figure(figsize=(12,7))
    ax2.plot(df_ts2['y'], 'green', color='blue', label='Training Data')
    ax2.plot(test_data2.index, predictions2, color='green', marker='o', linestyle='dashed', 
            label='Predicted')
    ax2.plot(test_data2.index, test_data2['y'], color='red', label='Actual')
    ax2.set_title('Rainfall Prediction')
    ax2.set_xlabel('Dates')
    ax2.set_ylabel('mm/day')
    st.pyplot(fig2)


elif choice == 'New Prediction By Day':
    d1 = st.date_input("When is the date you want to predict?", datetime(2019, 7, 6), min_value=date.today())
    st.write('The date you want to predict is:', d1)
    d0_d = df_ts1.index[-1]
    d0_d = d0_d.date()
    delta1 = d1 - d0_d
    delta1 = delta1.days
    # st.write(delta1)

    history1_d = history1
    predictions1_d = list()
    for t in range(delta1):
        model1_d = ARIMA(history1_d, order=(2,1,0)) 
        model_fit1_d = model1_d.fit(disp=0)
        output1_d = model_fit1_d.forecast()
        yhat1_d = output1_d[0]
        predictions1_d.append(yhat1_d)
        obs1_d = test_ar1[t]
        history1_d.append(obs1_d)

    st.write('The rainfall on', str(d1) ,'is:', round(predictions1_d[-1][0],2),'(mm/day-1)')

elif choice == 'New Prediction By Month':
    Years = np.arange(int(date.today().year),int(date.today().year)+11)
    # year = st.selectbox('Year',options=Years)
    Months = np.arange(1,13)
    # month = st.selectbox('Month',options=Months)
    # d2 = datetime(year, month, 1)
    # st.write('The month you want to predict is:', d2.date())
    with st.form("my_form"):
        st.write("Inside the form")
        # slider_val = st.slider("Form slider")
        # checkbox_val = st.checkbox("Form checkbox")
        # Every form must have a submit button.
        year = st.selectbox('Year',options=Years)
        month = st.selectbox('Month',options=Months)
        submitted = st.form_submit_button("Submit")
        d2 = datetime(year, month, 1)
        d2 = d2.date()
        if submitted:
            st.write('The month you want to predict is:', d2)
    st.write("Outside the form")
    d0_m = df_ts2.index[-1]
    d0_m = d0_m.date()
    delta2 = d2 - d0_m
    delta2 = math.floor(delta2.days/30)
    st.write(delta2)

    data_2020_d = df_ts.loc[(df_ts['ds'] >= '2020-01-01') & (df_ts['ds'] <= '2020-12-01')]
    data_2020_d.index = pd.to_datetime(data_2020_d.ds)
    data_2020_d = data_2020_d.drop(['ds'],axis=1)
    data_2020_m = data_2020_d.resample('MS').mean()

    # history2_m = history2
    # predictions2_m = predictions2
    # test_ar2m = test_ar2

    max_month = 12
    x = year - df_ts2.index[-1].year
    for t in range(delta2):
        index = df_ts2.index[-1].month + t
        if df_ts2.index[-1].month + t >= max_month:
            index = (df_ts2.index[-1].month + t) - max_month*math.floor(index/max_month)
        obs2_m = data_2020_m.values[index][0]
        history2.append(obs2_m)

        model2_m = ARIMA(history2, order=(2,1,0)) 
        model_fit2_m = model2_m.fit(disp=0)
        output2_m = model_fit2_m.forecast()
        yhat2_m = output2_m[0]

        predictions2.append(yhat2_m)
        st.write(yhat2_m)
    st.write('The rainfall on', str(d2) ,'is:', round(predictions2[-1][0],2),'(mm/month)')
