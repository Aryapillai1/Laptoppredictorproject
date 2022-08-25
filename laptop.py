
import pickle
import math
import numpy as np
import streamlit as st

pipe = pickle.load(open('pipe.pkl','rb'))
data = pickle.load(open('data.pkl','rb'))

st.title("LAPTOP PRICE  PREDICTOR")
st.header('Know your favourite laptops Price !')
##brand
company = st.selectbox('Brand you like',data['Company'].unique())
type = st.selectbox("Type of laptop you prefer",data['TypeName'].unique())
ram=st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])
weight=st.number_input('Weight of laptop')
touchscreen=st.selectbox('Touchscreen',['No','Yes'])
ips=st.selectbox('IPS',['No','Yes'])
screensize=st.number_input('Screen size')
resolution=st.selectbox('ScreenResolution',['1920 x 1080','1366x768','1600x900','3840 x 2160','3200 x 1800','2880 x 1800','2560 x 1440','2304 x 1440'])
cpu=st.selectbox('CPU',data['CpuBrand'].unique())
hdd=st.selectbox('HDD (in GB)',[0,128,256,512,1024,2048])
ssd=st.selectbox('SSD (in GB)',[0,8,128,256,512,1024])
gpu=st.selectbox('GPU',data['GpuBrand'].unique())
os=st.selectbox('OS',data['Opsys'].unique())
if st.button(' Predict Price'):
    ppi=None
    if touchscreen=='Yes':
        touchscreen=1
    else:
        touchscreen=0
    if ips=="Yes":
        ips=1
    else:
        ips=0

    X_res=int(resolution.split('x')[0])
    Y_res=int(resolution.split('x')[1])
    ppi=((X_res**2)+(Y_res**2))**0.5/screensize
    query=np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query=query.reshape(1,12)
    st.title("The predicted price is Rs.")
    st.title(int(np.exp(pipe.predict(query)[0])))






