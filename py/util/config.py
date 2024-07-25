import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Firebase credentials path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the JSON file
json_path = os.path.join(current_dir, '../../intern-b54ae-firebase-adminsdk-f0340-bc43dd3fdf.json')

FIREBASE_CREDENTIALS = json_path
