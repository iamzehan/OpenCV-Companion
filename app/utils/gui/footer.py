import json
import streamlit as st
import datetime as dt
import requests
import toml

def update_config(data):
    # Write data to the JSON file
    with open('.streamlit/secrets.toml', 'w') as toml_file:
        toml.dump(data, toml_file)
        
def get_latest_release(config):

    # GitHub API endpoint for releases
    api_url = config['footer']['latest_release_url']
    # Fetch the latest release information
    response = requests.get(api_url)
    latest_release_data = response.json()
    try:
        latest_release_version = latest_release_data["tag_name"]
        if latest_release_version!=config["footer"]["latest_release_url"]:
            config["footer"]["latest_release"]=latest_release_version
            update_config(config)
            return latest_release_version
    except:
        return config["footer"]["latest_release"]

def footer():
    config = toml.load(open(".streamlit/secrets.toml", 'r'))
    latest_release_version = get_latest_release(config)
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
                <p style="color:grey"> 
                <a href="{config["footer"]["linkedin"]}" target="_blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="{config["footer"]["linkedin"]}" height="20" width="30" /></a>
                <a href="{config["footer"]["github"]}" target="_blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/github.svg" height="20" width="30" /></a>
                <a href="{config["footer"]["mail"]}" target="_blank"><img align="center" src="https://upload.wikimedia.org/wikipedia/commons/7/7e/Gmail_icon_%282020%29.svg" height="15" width="30" /></a>
                <br>
                {latest_release_version}
                <br>
                &copy; {dt.datetime.now().year} Made by - <a href= "{config["footer"]["website"]}" target="_blank"> Md. Ziaul Karim </a>
            </footer>
            
            """,
            unsafe_allow_html=True
        )