import streamlit as st
from multiapp import MultiApp
from apps import main

app = MultiApp()

app.add_app("Upload", main.app)

app.run()
