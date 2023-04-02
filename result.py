import pickle 
import streamlit as st
loaded_model=pickle.load(open("C:/Users/VASA BHANU PRAKASH/BHANU/model10.sav",'rb'))
def prediction(date): 
    w=d[d==date].index[0] 
    print('Walmart sales predicting of given week:',end=' ') 
    print(sale_pred[w])
def main():
    st.title('WALMART SALES FORECASTING')
    sale=st.number_input('Walmart sales predicting of given week:')
    result=''
    if st.button('Final Prediction'):
        result=prediction(date)
    st.success(result)
    if __name__=='__main___':
       main()