import toml
import streamlit as st
import datetime as dt
import requests

def get_latest_release(config):

    # GitHub API endpoint for releases
    api_url = config['footer']['latest_release_url']
    
    try:
        # Fetch the latest release information
        response = requests.get(api_url)
        latest_release_data = response.json()
        latest_release_version = latest_release_data["tag_name"]
        return latest_release_version
    except:
        return "Latest"

def footer():
    config = toml.load(open(".streamlit/secrets.toml", 'r'))
    latest_release_version = get_latest_release(config)
    st.markdown(
            f"""
            ___
            
            <style>
                footer {{
                    color: grey;
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
                    color: #34eb5b;
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
            
            .Label--success {{
                    border-color: #34eb5b;
                    color: #34eb5b;
            }}

            .Label, .label {{
                border: 1px solid #34eb5b;
                border-radius: 1em;
                display: inline-block;
                font-size: var(--text-body-size-small, 0.75rem);
                font-weight: var(--base-text-weight-medium, 500);
                line-height: 18px;
                padding: 0 7px;
                white-space: nowrap;
            }}
            
            </style>
            <footer>
                <p style="color:grey"> 
                <a href="{config["footer"]["linkedin"]}" target="_blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="{config["footer"]["linkedin"]}" height="20" width="30" /></a>
                <a href="{config["footer"]["github"]}" target="_blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/github.svg" height="20" width="30" /></a>
                <a href="{config["footer"]["mail"]}" target="_blank"><img align="center" src="https://upload.wikimedia.org/wikipedia/commons/7/7e/Gmail_icon_%282020%29.svg" height="15" width="30" /></a>
                <div>
                    <b>Release:</b> <span title="Label: Latest" data-view-component="true" class="Label Label--success flex-shrink-0">
                        {latest_release_version}
                    </span>      
                </div>
                <div>
                &copy; {dt.datetime.now().year} Made by - <a href= "{config["footer"]["website"]}" target="_blank" style='color: #34eb5b'> Md. Ziaul Karim </a>
                </div>
            </footer>
            
            """,
            unsafe_allow_html=True
        )