import requests
import json
import time
import subprocess
import pandas as pd

def get_access_token():
    """Get the access token from the saved JSON file
    
    Returns:
        str: Access token for Spotify API
    """
    try:
        with open('spotify_token.json', 'r') as token_file:
            tokens = json.load(token_file)
            return tokens.get('access_token')
    except FileNotFoundError:
        print("Token file not found. Ensure you have completed the authorization.")
        return None

def get_all_saved_tracks(access_token):
    """Get all saved tracks for the user
    
    Args:
        access_token (str): Access token for Spotify API

    Returns:
        list: List of all saved tracks
    """
    url = "https://api.spotify.com/v1/me/top/tracks"
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    all_tracks = []
    offset = 0
    limit = 50  # Maximum limit allowed by Spotify API

    while True:
        params = {
            'type': 'tracks',
            'time_range': 'long_term',
            'limit': limit,
            'offset': offset
        }
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break

        data = response.json()
        items = data.get('items', [])
        if not items:
            break
        
        # Collect track information
        for i, track in enumerate(items):
            all_tracks.append({
                'track_id': track['id'],
                'track_name': track['name'],
                'artist': track['artists'][0]['name'],
                'explicit': track['explicit'],
                'popularity': track['popularity'],
                'album_name': track['album']['name']
            })
        
        # Increment offset for next batch
        offset += limit
        print(f"Fetched {len(all_tracks)} tracks so far...")

        # Avoid hitting rate limits
        time.sleep(0.1)

    return all_tracks

def get_audio_features(access_token, track_ids):
    """Get audio features for multiple tracks
    
    Args:
        access_token (str): Access token for Spotify API
        track_ids (list): List of track IDs

    Returns:
        list: List of audio features for the tracks
    """
    url = "https://api.spotify.com/v1/audio-features"
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    audio_features = []

    # Fetch in batches of 100 (Spotify API limit)
    for i in range(0, len(track_ids), 100):
        batch_ids = track_ids[i:i + 100]
        params = {
            'ids': ','.join(batch_ids)
        }
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break

        data = response.json()
        items = data.get('audio_features', [])
        
        # Collect track information
        for i, track in enumerate(items):
            if track is None:
                continue
            audio_features.append({
                'track_id': track['id'],
                'duration_ms': track['duration_ms'], 
                'acousticness': track['acousticness'], 
                'danceability': track['danceability'], 
                'energy': track['energy'], 
                'instrumentalness': track['instrumentalness'], 
                'key': track['key'], 
                'liveness': track['liveness'], 
                'loudness': track['loudness'], 
                'mode': track['mode'], 
                'speechiness': track['speechiness'], 
                'tempo': track['tempo'], 
                'valence': track['valence']
            })
        
        # Avoid hitting rate limits
        time.sleep(0.1)
    
    return audio_features

def fetch_tracks(access_token):
    """Fetch all saved tracks and their audio features
    
    Args:
        access_token (str): Access token for Spotify API

    Returns:
        pd.DataFrame: DataFrame of all saved tracks with audio features
    """
    if access_token:
        print("Fetching all saved tracks...")
        all_tracks = get_all_saved_tracks(access_token)
        
        if all_tracks:
            # Convert track data to a DataFrame
            df_tracks = pd.DataFrame(all_tracks)
            
            # Extract track IDs
            track_ids = df_tracks['track_id'].tolist()
            
            # Fetch audio features for all tracks
            print("Fetching audio features for tracks...")
            audio_features = get_audio_features(access_token, track_ids)
            df_audio_features = pd.DataFrame(audio_features)

            # Merge track data with audio features
            df = pd.merge(df_tracks, df_audio_features, left_on='track_id', right_on='track_id', how='inner')
            df['rank'] = range(1, len(df_audio_features) + 1)
            df['explicit'] = df['explicit'].astype(int)
            
            # Reorder columns for readability
            columns_order = ['rank', 'track_id', 'track_name', 'artist', 'album_name', 
                             'duration_ms', 'explicit', 'popularity', 'acousticness', 
                             'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 
                             'loudness', 'mode', 'speechiness', 'tempo', 'valence']
            df = df[columns_order]

            # Save the DataFrame to a JSON file
            # df.to_json('tracks.json', orient='records', lines=False)
            # Display the DataFrame
            print("All Tracks with Audio Features:")
            print(df)
        else:
            print("No tracks found or failed to fetch tracks.")
    else:
        print("Access token is missing. Unable to proceed.")

    return df

# No longer used in the final implementation due to API changes eliminating access to Audio Features
def search_song(search_term, access_token):
    """Search for a song and retrieve its audio features
    
    Args:
        search_term (str): Search term for the song
        access_token (str): Access token for Spotify API

    Returns:
        pd.DataFrame: DataFrame of the inputted song with audio features
    """
    track = {}

    url = "https://api.spotify.com/v1/search"
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    params = {
        'q': search_term,
        'type': 'track',
        'limit': 1,
    }
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")

    data = response.json()
    item = data.get('tracks', {}).get('items', [{}])[0]
    name = item.get('name')
    id = item.get('id')

    track['id'] = id
    track['track_name'] = name
    track['artist'] = item.get('artists', [{}])[0].get('name')
    track['album_name'] = item.get('album', {}).get('name')
    track['explicit'] = item.get('explicit')
    track['popularity'] = item.get('popularity')

    # Fetch audio features for the inputted song
    url = f"https://api.spotify.com/v1/audio-features/{id}"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")

    data = response.json()
    track['duration_ms'] = data.get('duration_ms')
    track['acousticness'] = data.get('acousticness')
    track['danceability'] = data.get('danceability')
    track['energy'] = data.get('energy')
    track['instrumentalness'] = data.get('instrumentalness')
    track['key'] = data.get('key')
    track['liveness'] = data.get('liveness')
    track['loudness'] = data.get('loudness')
    track['mode'] = data.get('mode')
    track['speechiness'] = data.get('speechiness')
    track['tempo'] = data.get('tempo')
    track['valence'] = data.get('valence')
    
    df_new_song = pd.DataFrame([track])

    print("Inputted Track with Audio Features:")
    print(df_new_song)

    return df_new_song

def search_song_v2(search_term, access_token):
    """Search for a song and retrieve its id
    
    Args:
        search_term (str): Search term for the song
        access_token (str): Access token for Spotify API

    Returns:
        str: ID of the inputted song
    """
    url = "https://api.spotify.com/v1/search"
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    params = {
        'q': search_term,
        'type': 'track',
        'limit': 2,
    }
    response = requests.get(url, headers=headers, params=params)

    data = response.json()
    item = data.get('tracks', {}).get('items', [{}])[0]
    name = item.get('name')
    id = item.get('id')

    print(f"Inputted Track: {name}")

    return id

def flask_server():
    # Step 1: Start Flask server for Spotify authentication
    flask_process = subprocess.Popen(['python', 'spotify-login.py'])

    # Step 2: Prompt the user to authorize the app
    print("Please go to http://localhost:3000/login to authorize the app.")
    print("After authorizing, press Enter to continue...")
    input() # Wait for user input after authorization
    
    # Step 3: Fetch the access token
    access_token = get_access_token()
    if access_token:
        print("Access token obtained successfully.")
    else:
        print("Failed to obtain access token.")

    # Step 4: Terminate the Flask server once tracks are returned
    flask_process.terminate()

    return access_token

if __name__ == "__main__":
    # Start the Flask server for Spotify authentication
    flask_server()

    # Fetch all saved tracks and their audio features
    access_token = get_access_token()
    fetch_tracks(access_token)

    # Search for a song and retrieve its audio features
    # search_song("Shape of You", access_token)

    # Search for a song and retrieve its id
    # search_song_v2("Shape of You", access_token)