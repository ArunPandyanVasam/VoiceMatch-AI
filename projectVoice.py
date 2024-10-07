import librosa
import sqlite3
import numpy as np
from numpy.linalg import norm
import sounddevice as sd
from scipy.io.wavfile import write

# 1. Take Audio as User Input
sampling_frequency = 44100
recording_duration = 5
recording = sd.rec(int(recording_duration * sampling_frequency),
                   samplerate=sampling_frequency, channels=1)
sd.wait()
write("userRecording0.wav", sampling_frequency, recording.astype(np.int16))

# 2. The User Audio should be converted to MFCC Values 
#    and Database voices should also be converted to MFCC Values
def get_mfccValues(audio_file, n_mfcc=13):
    audio_data, sampling_rate = librosa.load(audio_file, sr=44100, mono=True)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=n_mfcc)
    return mfcc

# 3. Convert MFCC to 1-Dimensional Array
def flatten_mfcc(mfcc):
    return mfcc.flatten()

# 4. Function to pad or truncate the MFCC arrays
def pad_or_truncate(arr, target_length):
    if len(arr) < target_length:
        # Pad with zeros if the array is shorter
        return np.pad(arr, (0, target_length - len(arr)), 'constant')
    else:
        # Truncate if the array is longer
        return arr[:target_length]

# 5. Path for user audio
user_audio = '/home/avasam/Github/VoiceMatch AI/userRecording0.wav' 
mfcc_user = flatten_mfcc(get_mfccValues(user_audio))

# 6. Printing user MFCC Values
#   print("User MFCC Values:", mfcc_user)

# 10. Function to calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# 11. Retrieve MFCC values from the database and compare with user's MFCC
with sqlite3.connect('voicedatabase.db') as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT audio_file, mfcc FROM mfcc_values")
    rows = cursor.fetchall()

    similarities = []

    # Define a threshold for low similarity (you can adjust this value)
    low_similarity_threshold = 0.5  # This threshold can be adjusted based on your needs

    for audio_file, mfcc_blob in rows:
        mfcc_database = np.frombuffer(mfcc_blob, dtype=np.float32)
        
        # Pad or truncate the user MFCC to match the database MFCC
        padded_mfcc_user = pad_or_truncate(mfcc_user, len(mfcc_database))
        
        similarity_score = cosine_similarity(padded_mfcc_user, mfcc_database)
        
        # Scale the score to a percentage from 0 to 100
        percentage_score = ((similarity_score + 1) / 2) * 100
        
        # If the score is below the threshold, set it to be between 0 and 50
        if similarity_score < low_similarity_threshold:
            percentage_score = (percentage_score / 100) * 50  # Scale down to 0-50 range
        
        similarities.append((audio_file, percentage_score))

# 12. Find the audio file with the highest similarity percentage score
most_similar_audio = max(similarities, key=lambda x: x[1])

print(f"The audio file most similar to the user recording is '{most_similar_audio[0]}' with a similarity score of {most_similar_audio[1]:.2f}%.")
