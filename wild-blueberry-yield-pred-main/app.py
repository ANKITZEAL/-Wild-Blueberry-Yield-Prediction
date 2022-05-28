import streamlit as st
from model import predict_yield
import numpy as np

st.set_page_config(page_title="Wild Blueberry Yield Prediction App",
                   page_icon="🍇", layout="wide")

col1, col2 = st.beta_columns([3,5])
# ['clonesize', 'honeybee', 'osmia', 'MinOfUpperTRange', 
#        'MaxOfLowerTRange', 'RainingDays', 'AverageRainingDays', 'fruitset',
#        'fruitmass', 'seeds']
with col1:

    with st.form("prediction_form"):

        st.header("Enter the Deciding Factors:")

        clonesize = st.text_input("clonesize value")
        honeybee = st.text_input("honeybee density value")
        osmia = st.text_input("osmia density value")
        MinOfUpperTRange = st.text_input("min-upper temperature Range")
        MaxOfLowerTRange = st.text_input("max-lower temperature Range")
        RainingDays = st.text_input("raning days value")
        AverageRainingDays = st.text_input("average raining days value")
        fruitset = st.text_input("fruitset value")
        fruitmass = st.text_input("fruitmass value")
        seeds = st.text_input("seeds value")

        submit_val = st.form_submit_button("Predict Yield")

if submit_val:
    print(clonesize)
    attribute = np.array([float(clonesize), float(honeybee), float(osmia),
                        float(MinOfUpperTRange), float(MaxOfLowerTRange),
                        float(RainingDays), float(AverageRainingDays),
                        float(fruitset), float(fruitmass), float(seeds)]).reshape(1,-1)


    if attribute.shape == (1,10):
        print("attrubutes valid")
        
        value, img = predict_yield(attributes= attribute)

        with col2:
            st.header("Here are the results:")
            st.success(f"The yield value is {value}")
            st.image(img, caption="Yield behaviour")
            