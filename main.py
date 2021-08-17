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
from datetime import date, datetime, timedelta
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
    obs1 = test_ar1[t]
    history1.append(obs1)
    model1 = ARIMA(history1, order=(0,1,0)) 
    model_fit1 = model1.fit(disp=0)
    output1 = model_fit1.forecast()
    yhat1 = output1[0]
    predictions1.append(yhat1)
mse1 = mean_squared_error(test_ar1, predictions1)
mae1=mean_absolute_error(test_ar1, predictions1)
rmse1 = np.sqrt(mse1) 

# Build Model theo thang
train_data2, test_data2 = df_ts2[0:int(len(df_ts2)*0.8)], df_ts2[int(len(df_ts2)*0.8):]
train_ar2 = train_data2['y'].values
test_ar2 = test_data2['y'].values

history2 = [x for x in train_ar2]
predictions2 = list()
for t in range(len(test_ar2)):
    obs2 = test_ar2[t]
    history2.append(obs2)
    model2 = ARIMA(history2, order=(0,1,0)) 
    model_fit2 = model2.fit(disp=0)
    output2 = model_fit2.forecast()
    yhat2 = output2[0]
    predictions2.append(yhat2)
    # print('predicted=%f, expected=%f' % (yhat, obs))
mse2 = mean_squared_error(test_ar2, predictions2)
mae2 = mean_absolute_error(test_ar2, predictions2)
rmse2 = np.sqrt(mse2) 

# Part 2: Show project's result with Streamlit

st.title("Data Science")
st.header('Rainfall Prediction Project')
st.subheader('Make new Prediction')

menu = ['Overview','Build Project','New Prediction By Day','New Prediction By Month']
choice = st.sidebar.selectbox('Menu',menu)
if choice == 'Overview':
    st.subheader('Overview')
    st.write("""
    Thành phố Hồ Chí Minh nằm trong vùng nhiệt đới gió mùa cận xích đạo. Cũng như các tỉnh ở Nam bộ, đặc điểm chung của khí hậu-thời tiết TPHCM là nhiệt độ cao đều trong năm và có hai mùa mưa - khô rõ ràng làm tác động chi phối môi trường cảnh quan sâu sắc. Mùa mưa từ tháng 5 đến tháng 11, mùa khô từ tháng 12 đến tháng 4 năm sau. Xét yếu tố lượng mưa, có hơn 90% lượng mưa tập trung vào tháng 5 đến tháng 11. Hiện nay đã có sự biến thiên về lượng mưa, tạo ra ngập cục bộ. Các dự báo cho thấy rằng tổng lượng mưa hàng năm có xu hướng tăng dần tuy không nhiều nhưng sẽ có sự biến thiên giữa các mùa lớn hơn. Ngập cục bộ vì vậy sẽ tăng và mưa lớn kết hợp với bão sẽ trở nên phổ biến hơn. Ở đây ta phải hiểu rằng mưa nhiều không phải là nguyên nhân chính gây ra ngập ở TPHCM vì lưu vực TPHCM đã được quản lý chặc chẻ, nhưng ngập cục bộ từ một trận mưa cường độ cao sẽ là một mối đe dọa lớn.

    Ngập thường xuyên có thể gây sự phát tán các chất ô nhiễm nghiêm trọng và gây ra mối đe dọa với sức khỏe cộng đồng, sản xuất kinh tế, và các hệ sinh thái. Bên cạnh đó mưa nhiều cũng ảnh trực tiếp đến một số ngành nghề nhất định như xây dựng, du lịch, vận tải…

    Chính vì vậy với mục đích cung cấp và phổ biến thông tin về lượng mưa đến tất cả mọi người, giúp mỗi người có cái nhìn khách quan về hiện trạng mưa ở thành phố, tác giả sẽ xây dựng các mô hình dự đoán lượng mưa theo thời gian trong tương lai dựa trên dữ liệu đã thu thập được trong quá khứ.
    """)
elif choice == 'Build Project':
    st.subheader('Build Project')
    st.write('### Data preprocessing')
    st.write('### Show data:')
    st.table(df_ts.head(5))

    st.write('### Build model and evaluation with data by date:')
    
    fig1,ax1 = plt.subplots()
    # ax1.figure(figsize=(12,7))
    ax1.plot(df_ts1['y'], 'green', color='blue', label='Training Data')
    ax1.plot(test_data1.index, predictions1, color='green', marker='o', linestyle='dashed', 
            label='Predicted')
    ax1.plot(test_data1.index, test_data1['y'], color='red', label='Actual')
    ax1.set_title('Rainfall Prediction')
    ax1.set_xlabel('Dates')
    ax1.set_ylabel('mm/day')
    ax1.legend()
    st.pyplot(fig1)
    
    st.write('Mean square error (date): {}'.format (round(mse1,4)))
    st.write('Root mean square error (date): {}'.format (round(rmse1,4)))
    st.write('Mean absolute error (date): {}'.format (round(mae1,4)))

    st.write('### Build model and evaluation with data by month:')
    
    fig2,ax2 = plt.subplots()
    # ax2.figure(figsize=(12,7))
    ax2.plot(df_ts2['y'], 'green', color='blue', label='Training Data')
    ax2.plot(test_data2.index, predictions2, color='green', marker='o', linestyle='dashed', 
            label='Predicted')
    ax2.plot(test_data2.index, test_data2['y'], color='red', label='Actual')
    ax2.set_title('Rainfall Prediction')
    ax2.set_xlabel('Dates')
    ax2.set_ylabel('mm/day')
    ax2.legend()
    st.pyplot(fig2)
    
    st.write('Mean square error (month): {}'.format (round(mse2,4)))
    st.write('Root mean square error (month): {}'.format (round(rmse2,4)))
    st.write('Mean absolute error (month): {}'.format (round(mae2,4)))

elif choice == 'New Prediction By Day':
    d1 = st.date_input("When is the date you want to predict?",date.today() , min_value=date.today())
    st.write('The date you want to predict is:', d1) # datetime(2019, 7, 6)
    d0_d = df_ts1.index[-1]
    d0_d = d0_d.date()
    delta1 = d1 - d0_d
    delta1 = delta1.days

    data_last_year_d = df_ts.loc[(df_ts['ds'] >= '2020-01-01') & (df_ts['ds'] <= '2020-12-31')]
    data_last_year_d.index = pd.to_datetime(data_last_year_d.ds)
    data_last_year_d = data_last_year_d.drop(['ds'],axis=1)

    dict_day = {}
    for i in range(len(data_last_year_d)):
        dict_day[str(data_last_year_d.index[i].month) + str(data_last_year_d.index[i].day)] = data_last_year_d.values[i][0]


    for t in range(delta1):
        new_date = d0_d + timedelta(t + 1)
        key = str(new_date.month) + str(new_date.day)
        
        obs2_m = dict_day[key]
        history1.append(obs2_m)

        model1_d = ARIMA(history1, order=(0,1,0)) 
        model_fit1_d = model1_d.fit(disp=0)
        output1_d = model_fit1_d.forecast()
        yhat1_d = output1_d[0]
        if(yhat1_d[0] < 0):
            yhat1_d[0] = 0
        predictions1.append(yhat1_d)
        dict_day[key] = yhat1_d[0]

    st.write('The rainfall on', str(d1) ,'is:', round(predictions1[-1][0],2),'(mm/day-1)')

elif choice == 'New Prediction By Month':
    Years = np.arange(int(date.today().year),int(date.today().year)+11)
    Months = np.arange(1,13)
    flag = -1
    with st.form("my_form"):
        st.write("Enter information")
        year = st.selectbox('Year',options=Years)
        month = st.selectbox('Month',options=Months)

        # if (year == date.today().year):
        #     month = st.selectbox('Month',options=np.arange(int(date.today().month),13))
        # else:
        #     month = st.selectbox('Month',options=Months)

        submitted = st.form_submit_button("Submit")
        d2 = datetime(year, month, 1)
        d2 = d2.date()
        if submitted:
            st.write('The month you want to predict is:', d2)
    if (d2.year <= date.today().year) & (d2.month < date.today().month):
        st.write('Please select future prediction time')
        flag = 0
    else:
        flag = 1
    if flag == 1:
        st.write("Result")
        d0_m = df_ts2.index[-1]
        d0_m = d0_m.date()
        delta2 = d2 - d0_m
        delta2 = math.floor(delta2.days/30)
        # st.write(delta2)

        data_last_year_d = df_ts.loc[(df_ts['ds'] >= '2020-01-01') & (df_ts['ds'] <= '2020-12-01')]
        data_last_year_d.index = pd.to_datetime(data_last_year_d.ds)
        data_last_year_d = data_last_year_d.drop(['ds'],axis=1)
        data_last_year_m = data_last_year_d.resample('MS').mean()

        dict_month = {}
        for i in range(len(data_last_year_m)):
            dict_month[data_last_year_m.index[i].month] = data_last_year_m.values[i][0]


        max_month = 12
        x = year - df_ts2.index[-1].year
        for t in range(delta2):
            index = df_ts2.index[-1].month + t
            if index >= max_month:
                index = index - max_month*math.floor(index/max_month)
            
            obs2_m = dict_month[index+1]
            history2.append(obs2_m)

            model2_m = ARIMA(history2, order=(0,1,0)) 
            model_fit2_m = model2_m.fit(disp=0)
            output2_m = model_fit2_m.forecast()
            yhat2_m = output2_m[0]

            if(yhat2_m[0] < 0):
                yhat2_m[0] = 0
            predictions2.append(yhat2_m)

            dict_month[index+1] = yhat2_m[0]
        st.write('The rainfall on', str(d2) ,'is:', round(predictions2[-1][0],2),'(mm/month)')
