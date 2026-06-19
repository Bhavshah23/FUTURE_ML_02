import streamlit as st

from predict import predict_ticket

st.title("Support Ticket Intelligence System")

ticket = st.text_area("Enter Support Ticket")

if st.button("Predict"):

    category, priority = predict_ticket(ticket)

    st.success(f"Category: {category}")

    st.warning(f"Priority: {priority}")