import json
import streamlit as st
import datetime as dt

def footer():
    config = json.load(open("config.json"))
    st.markdown(
            f"""
            ___
            
            <style>
                footer {{
                    color: #fff;
                    text-align: center;
                    padding: 1rem;
                    position: relative;
                    bottom: 0;
                    width: 100%;
                }}
                a {{
                    color: #fff;
                    text-decoration: none;
                    font-weight: bold;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
                
                .blank-div {{
                background-color: black;
                width: 100%;
                height: 480px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white; /* set the text color to white for better visibility */
                font-size: 24px;
            }}
            </style>
            <footer>
                <a href="{config["Linkedin"]}" target="_blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="{config["Linkedin"]}" height="20" width="30" /></a>
                <a href="{config["Github"]}" target="_blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/github.svg" height="20" width="30" /></a>
                <a href="https://mail.google.com/mail/?view=cm&fs=1&to=ziaul.karim497@gmail.com" target="_blank"><img align="center" src="https://upload.wikimedia.org/wikipedia/commons/7/7e/Gmail_icon_%282020%29.svg" height="15" width="30" /></a>
                <br>
                &copy; {dt.datetime.now().year} Made by - <a href= "https://ziaulkarim.netlify.app/" target="_blank"> Md. Ziaul Karim </a>
            </footer>
            """,
            unsafe_allow_html=True
        )