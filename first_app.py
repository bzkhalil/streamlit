import streamlit as st
from pytube import YouTube

import numpy as np
import pandas as pd
st.title("Youtube Video Donwloader")
st.subheader("Enter the URL:")
url = st.text_input(label='URL')
