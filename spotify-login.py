from config import CLIENT_ID, CLIENT_SECRET
import os
import requests
import random
import string
import base64
import json
from flask import Flask, redirect, request, session, url_for, jsonify

# Flask app setup
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Spotify API credentials
REDIRECT_URI = 'http://localhost:3000/callback'
SCOPE = 'user-read-private user-read-email user-top-read'
STATE_KEY = 'spotify_auth_state'

# Function to generate a random state string
def generate_random_state(length=16):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Step 1: Request User Authorization
@app.route('/login')
def login():
    state = generate_random_state()
    session[STATE_KEY] = state
    auth_url = 'https://accounts.spotify.com/authorize'
    params = {
        'response_type': 'code',
        'client_id': CLIENT_ID,
        'scope': SCOPE,
        'redirect_uri': REDIRECT_URI,
        'state': state,
        'show_dialog': 'true'
    }
    return redirect(f"{auth_url}?{requests.compat.urlencode(params)}")

# Step 2: Spotify Callback
@app.route('/callback')
def callback():
    # Retrieve the code and state from the query parameters
    code = request.args.get('code')
    state = request.args.get('state')

    # Check if state matches to prevent CSRF attacks
    if state != session.get(STATE_KEY):
        return "State mismatch. Potential CSRF attack.", 400
    
    # Combine client_id and client_secret with a colon
    client_credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"

    # Encode the string using base64
    encoded_credentials = base64.b64encode(client_credentials.encode()).decode()

    # Step 3: Exchange Authorization Code for Access Token
    token_url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': 'Basic ' + encoded_credentials,
    }

    data = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': REDIRECT_URI
    }
    
    response = requests.post(token_url, headers=headers, data=data)
    if response.status_code != 200:
        return f"Failed to get token: {response.status_code} {response.text}", 400

    tokens = response.json()
    access_token = tokens.get('access_token')
    refresh_token = tokens.get('refresh_token')

    # Save the access token to a JSON file
    with open('spotify_token.json', 'w') as token_file:
        json.dump({'access_token': access_token, 'refresh_token': refresh_token}, token_file)

    # Step 4: Use Access Token to Get User Profile
    user_profile = get_spotify_user_profile(access_token)
    return jsonify(user_profile)

# Helper function to get Spotify user profile using the access token
def get_spotify_user_profile(access_token):
    profile_url = "https://api.spotify.com/v1/me"
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get(profile_url, headers=headers)
    return response.json()

if __name__ == '__main__':
    app.run(port=3000, debug=False)
