import streamlit as st

st.set_page_config(page_title="PPK Processing", layout="wide")

st.title("PPK Processing Tool")
st.markdown("""
Welcome to the **PPK Processing Tool**.  
This app will eventually handle your RINEX, base station, and observation files  
to generate accurate EXIF geotagging for photogrammetry workflows.
""")

uploaded_file = st.file_uploader("Upload a sample file", type=["txt", "csv", "bin", "obs", "nav"])

if uploaded_file:
    st.success(f"File `{uploaded_file.name}` uploaded successfully!")
    st.write("File details:")
    st.write(f"- Size: {len(uploaded_file.getvalue())} bytes")
    st.write(f"- Type: {uploaded_file.type}")
else:
    st.info("👆 Upload a file to begin.")
