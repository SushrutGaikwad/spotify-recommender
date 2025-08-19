import requests
import time

APP_URL = "http://localhost:8000"


# Getting the status code
def get_app_status(url):
    response = requests.get(url)
    status_code = response.status_code
    return status_code


# Testing for the app home page loading
def test_app_loading():
    # Waiting for the app to load
    time.sleep(60)
    status_code = get_app_status(APP_URL)
    assert status_code == 200, "Unable to load Streamlit App"
    print("Streamlit App Loaded Successfully")
