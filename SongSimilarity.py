import pandas as pd
import numpy as np

def song_similarity(song_data, new_song):
    """Return the most similar song to the inputted song
    
    Args:
        song_data (pd.DataFrame): DataFrame of all songs
        new_song (pd.DataFrame): DataFrame of the input
    
    Returns:
        tuple: Most similar song and artist
    """
    similarity_tracks = song_data.copy()

    # Find attributes of new song given by user
    features = ["acousticness", "danceability", "energy", "instrumentalness", "key", "liveness", "loudness", "mode", "speechiness", "tempo", "valence"]

    all_songs_feature_matrix = similarity_tracks[features].values
    new_song_feature_matrix = new_song[features].values.flatten()

    dot_product = np.dot(all_songs_feature_matrix, new_song_feature_matrix)
    magnitudes = np.linalg.norm(all_songs_feature_matrix, axis=1) * np.linalg.norm(new_song_feature_matrix)

    cosine_similarity = dot_product / magnitudes

    similarity_tracks["cosine_similarity"] = cosine_similarity
    sorted_similarity = similarity_tracks.sort_values(by="cosine_similarity", ascending=False)
    most_similar_song = sorted_similarity.iloc[0]

    return most_similar_song["track_name"], most_similar_song["artist"]

def most_similar_songs(training_tracks, test_tracks):
    """Find the most similar songs to the test songs
    
    Args:
        training_tracks (pd.DataFrame): DataFrame of training tracks
        test_tracks (pd.DataFrame): DataFrame of test
    """
    song_similarity_results = []

    for index, test_song in test_tracks.iterrows():
        most_similar_song, most_similar_song_artist = song_similarity(training_tracks, test_song)
        song_similarity_results.append({
            "song": test_song["track_name"],
            "artist": test_song["artist"],
            "most_similar_song": most_similar_song,
            "most_similar_song_artist": most_similar_song_artist
        })

    song_similarity_df = pd.DataFrame(song_similarity_results)
    
    print("Song Similarity Results:")
    print(song_similarity_df)

if __name__ == "__main__":
    tracks = pd.read_json("data/tracks.json")
    most_similar_songs(tracks, tracks.sample(5))