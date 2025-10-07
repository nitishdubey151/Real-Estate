import pickle

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Viz Demo")

with open('df.pkl', 'rb') as file:
    df = pickle.load(file)

with open('pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)


st.header('Enter your inputs')

#property_type
property_type =st.selectbox('Property Type',['flat','house'])

#sector
sector = st.selectbox('Sector',sorted(df['sector'].unique().tolist()))

bedroom = float(st.selectbox('No of Bedroom',sorted(df['bedRoom'].unique().tolist())))

bathroom = float(st.selectbox('No of Bathroom',sorted(df['bathroom'].unique().tolist())))

balcony = st.selectbox('Balconies',sorted(df['balcony'].unique().tolist()))

property_age = st.selectbox('Property Age',sorted(df['agePossession'].unique()))

built_up_area = float(st.number_input('Built_up_area'))

servent_room = float(st.selectbox('Servent Room',[0.0,1.0]))
store_room = float(st.selectbox('Store Room',[0.0,1.0]))

furnishing_type = st.selectbox('Furnishing Type',sorted(df['furnishing_type'].unique()))
luxury_category = st.selectbox('Luxury Category',sorted(df['luxury_category'].unique()))
floor_category = st.selectbox('Floor Category',sorted(df['floor_category'].unique()))

if st.button('Predict'):
    # form a dataframe
    data = [[property_type, sector, bedroom, bathroom, balcony, property_age, built_up_area, servent_room, store_room, furnishing_type, luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']

    # Convert to DataFrame
    one_df = pd.DataFrame(data, columns=columns)

    st.dataframe(one_df)

    # predict

    base_price = np.expm1(pipeline.predict(one_df))[0]
    low = base_price - 0.22
    high = base_price + 0.22

    #display
    st.text("The price of flat is between {} and {} cr".format(round(low,2), round(high,2)))






