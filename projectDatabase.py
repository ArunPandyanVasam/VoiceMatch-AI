import sqlite3
import numpy as np
import librosa

# 2. The User Audio should be converted to MFCC Values 
#    and Database voices should also be converted to MFCC Values
def get_mfccValues(audio_file, n_mfcc=13):
    audio_data, sampling_rate = librosa.load(audio_file, sr=44100, mono=True)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=n_mfcc)
    return mfcc

# 3. Convert MFCC to 1-Dimensional Array
def flatten_mfcc(mfcc):
    return mfcc.flatten()

# 7. Load and Extract MFCCs for the Database Audios
database_audios = [
    '/home/avasam/Github/VoiceMatch AI/DatabaseVoices/BlackPather.wav',
    '/home/avasam/Github/VoiceMatch AI/DatabaseVoices/BlackWidow.wav',
    '/home/avasam/Github/VoiceMatch AI/DatabaseVoices/HarryPotter.wav',
    '/home/avasam/Github/VoiceMatch AI/DatabaseVoices/IronMan.wav',
    '/home/avasam/Github/VoiceMatch AI/DatabaseVoices/Leonardo.wav',
    '/home/avasam/Github/VoiceMatch AI/DatabaseVoices/SpiderMan.wav',
    '/home/avasam/Github/VoiceMatch AI/DatabaseVoices/TomCruise.wav'
]

# 8. Connect to SQLite database (it will create one if it doesn't exist)
with sqlite3.connect('voicedatabase.db') as conn:
    cursor = conn.cursor()

    # 9. Create a table to store MFCCs if it doesn't already exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mfcc_values (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        audio_file TEXT,
        mfcc BLOB
    )
    ''')

    # Loop through each audio file, extract MFCC, and store in the database
    for audio_path in database_audios:
        # Load the audio file and compute MFCCs
        mfcc_database = flatten_mfcc(get_mfccValues(audio_path))

        # Convert MFCC to binary format
        mfcc_blob = mfcc_database.tobytes()  # Convert the array to bytes for storage

        # Extract the file name from the path
        audio_file_name = audio_path.split('/')[-1]

        # Insert MFCC values into the database
        cursor.execute('''
        INSERT INTO mfcc_values (audio_file, mfcc)
        VALUES (?, ?)
        ''', (audio_file_name, mfcc_blob))

print("MFCC values for all actors stored successfully.")