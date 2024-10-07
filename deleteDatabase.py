import os

# Specify the name of your database file
database_path = "voicedatabase.db"

# Check if the file exists and then delete it
if os.path.exists(database_path):
    os.remove(database_path)
    print("Database deleted successfully")
else:
    print("The database does not exist")
