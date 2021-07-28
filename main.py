import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from fbprophet import Prophet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import pandas_profiling as pp

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima

from fbprophet import Prophet 
from fbprophet.plot import add_changepoints_to_plot

data = pd.read_csv('data_rain.csv',skiprows=10)

data.head()

data['DATE'] = data['YEAR'].astype(str) + '-' + data['MO'].astype(str) + '-' + data['DY'].astype(str)
data.head()

df = data[['DATE','PRECTOT']]
df.head()

df = df.drop(df[df.PRECTOT <0].index)

df.tail()

df_ts = pd.DataFrame()
df_ts['ds'] = pd.to_datetime(df['DATE'])
df_ts['y'] = df['PRECTOT']

df_ts.head()

train = df_ts[df_ts['ds']<'2018-01-01']
train.head()

test = df_ts[df_ts['ds']>='2018-01-01']
test.head()

test.tail()

# Build Model
model = Prophet()
model.fit(df_ts)
# 2 year in test and 2 year to predict new values
months = pd.date_range('2018-01-01','2023-07-06',freq='D').strftime("%Y-%m-%d").tolist()    
future = pd.DataFrame(months)
future.columns = ['ds']
future['ds'] = pd.to_datetime(future['ds'])
forecast = model.predict(future)
forecast[['ds', 'yhat']].head()
y_test =test['y'].values
y_pred = forecast['yhat'].values[:1283]
mae_p = mean_absolute_error(y_test,y_pred)
print(mae_p)
y_test_value = pd.DataFrame(y_test, index = pd.to_datetime(test['ds']),columns=['Actual'])
y_pred_value = pd.DataFrame(y_pred, index = pd.to_datetime(test['ds']),columns=['Prediction'])

y_test_value

# Visulaize the result
plt.figure(figsize=(12,6))
plt.plot(y_test_value, label='Rainfall')
plt.plot(y_pred_value, label='Rainfall prediction')
plt.xticks(rotation='vertical')
plt.legend()
plt.show()

fig = model.plot(forecast) 
fig.show()
a = add_changepoints_to_plot(fig.gca(), model, forecast)

fig1 = model.plot_components(forecast)
fig1.show()

# Prediction for next 2 years
m = Prophet() 
m.fit(df_ts)
future = m.make_future_dataframe(periods=365, freq='D') # next 365 days

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']].tail()

fig = m.plot(forecast) 
fig.show()
a = add_changepoints_to_plot(fig.gca(), m, forecast)

fig1 = model.plot_components(forecast)
fig1.show()

plt.figure(figsize=(15,8))
plt.plot(df_ts['y'], label='Rainfall')
plt.plot(forecast['yhat'], label='Rainfall with next 30 days prediction', color='red')
plt.xticks(rotation='vertical')
plt.legend()
plt.show()

# Part 2: Show project's result with Streamlit

st.title("Data Science")
st.header('Rainfall Prediction Project')
st.subheader('Make new Prediction')

menu = ['Overview','Build Project','New Prediction']
choice = st.sidebar.selectbox('Danh muc',menu)
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

    st.write('### Build model and evaluation:')
    st.write('Mean absolute error: {}'.format (round(mae_p,2)))

    st.write('#### Visualization')
    plt.figure(figsize=(12,6))
    plt.plot(y_test_value, label='Rainfall')
    plt.plot(y_pred_value, label='Rainfall prediction')
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.show()

    fig = model.plot(forecast) 
    fig.show()
    a = add_changepoints_to_plot(fig.gca(), model, forecast)

    fig1 = model.plot_components(forecast)
    fig1.show()
