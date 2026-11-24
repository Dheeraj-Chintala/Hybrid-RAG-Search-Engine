import streamlit as st

st.html('<script>console.log("Streamlit custom script");</script>', unsafe_allow_javascript=True)
print("st.html OK!")
