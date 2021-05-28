import streamlit as st


def css_loader(file):
    """ Loads custom css file.
    :param file: css file location
    """
    with open(file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
