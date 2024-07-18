import pickle 
import streamlit as st

model1 = pickle.load(open("Task-1.pkl", "rb"))

def myf1():
    st.title("Iris Classification")
    a = st.number_input("Enter Sepal_length: ")
    b = st.number_input("Enter Sepal_width: ")
    c = st.number_input("Enter Petal_length: ")
    d = st.number_input("Enter Petal_width: ")
    pred = st.button("Classify")

    if  pred:
        op = model1.predict([[a, b, c, d]])
        st.write("Species is :",op[0])

        
     
myf1()