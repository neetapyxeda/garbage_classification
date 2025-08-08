import streamlit as st
import authlib
from app import run_app

#from app import chat

IMAGE_ADDRESS = "https://img.goodfon.com/original/1920x1080/8/71/fireman-fire-man.jpg"

# title
#st.title("Google Login App")

#st.image(IMAGE_ADDRESS)
#if not st.experimental_user.is_logged_in:

if not st.user.is_logged_in:
    st.title("Google Login App")
    st.image(IMAGE_ADDRESS)
    if st.sidebar.button("Log in with Google", type="primary", icon=":material/login:"):
        st.login()

else:
    #st.html(f"Hello, <span style='color: orange; font-weight: bold;'>{st.experimental_user.name}</span>!")
    if st.sidebar.button("Log out", type="secondary", icon=":material/logout:"):
        st.logout()
    #chat()

    run_app()

