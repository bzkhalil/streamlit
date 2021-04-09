
from streamlit_player import st_player
import streamlit as st



with st.sidebar:
    "## ⚙️ Parameters"

    options = {
        "events": st.multiselect("Events to listen", ["onProgress"], ["onProgress"]),
        "progress_interval": st.slider("Progress refresh interval (ms)", 200, 2000, 250, 1),
        "volume": st.slider("Volume", 0.0, 1.0, 1.0, .01),
        "playing": st.checkbox("Playing", False),
        "loop": st.checkbox("Loop", False),
        "controls": st.checkbox("Controls", True),
        "muted": st.checkbox("Muted", False),
    }


events = ["onProgress"]

#event = st_player('https://www.youtube.com/watch?v=oRQyu66zGE4',events=events,progress_interval=500)
#event = st_player('https://youtu.be/CmSKVW1v0xM',events=events,progress_interval=500)
url = 'https://youtu.be/CmSKVW1v0xM'
event = st_player(url, **options, key=1)
#t = st.progress(event.data['played'])
t = st.slider('',0.0,1.0,float(event.data['played']),0.01)
event


