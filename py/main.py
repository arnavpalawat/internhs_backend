import os
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, jsonify, request
import logging
from groq import Groq
from jobspy import scrape_jobs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from py.util import config
from py.util.config import FIREBASE_CREDENTIALS
from py.util.firebase import initialize_firebase
from py.util.jobs import Job, Jobs
from datetime import datetime
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS requests from any origin

# Change directory to repository root
repository_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
os.chdir(repository_root)

# Initialize Firebase Admin
initialize_firebase(FIREBASE_CREDENTIALS)


# Function to filter jobs based on keywords and phrases
def filter_jobs(jobs_list):
    print("Filtering jobs.")
    exclude_keywords = ["bachelor", "accredited", "college", "undergraduate", "major", "mba", "degree", "Fulltime",
                        "Full-time", "full time", "Full-Time", "full-time", "Full Time", "related experience",
                        "CEO", "ceo", "CFO", "cfo", " ms ", " MS", "PhD", "+ years", "Ph.D", "Pursuing a ",
                        "Graduate", "graduate", "diploma", "Diploma", "GED", "College", "Undergraduate", "18", "Degree"]
    include_keywords = ["intern", "high school"]
    include_phrases = ["high school", "highschool", "internship"]
    exclude_phrases = ["college", "university"]
    exclude_titles = ["CEO", "CFO", "CTO", "Chief", "Director", "Manager", "Senior"]

    filtered_jobs = []

    for job in jobs_list:
        description_lower = str(job.description).lower()
        title_lower = job.title.lower()

        if all(keyword not in description_lower for keyword in exclude_keywords) \
                and any(keyword in description_lower for keyword in include_keywords) \
                and any(phrase in description_lower for phrase in include_phrases) \
                and all(phrase not in description_lower for phrase in exclude_phrases) \
                and all(title not in title_lower for title in exclude_titles):
            filtered_jobs.append(job)

    print(f"Filtered {len(filtered_jobs)} jobs.")
    return filtered_jobs


# Function to scrape jobs and handle errors
def scrape_jobs_from_params(country, radius, remote, age):
    try:
        jobs = scrape_jobs(
            site_name=["indeed", "zip_recruiter", "glassdoor"],
            search_term="Intern",
            results_wanted=100,
            country_indeed=country,
            hours_old=age,
            distance=radius,
            is_remote=remote,
        )
        print("Jobs scraped successfully.")
        return jobs
    except Exception as e:
        print(f"Error scraping jobs: {e}")
        raise


# Function to get job prestige from Groq API
def get_job_prestige(filtered_jobs):
    final_job_list = []
    job_data = []
    api_key = config.GROQ_API_KEY
    client = Groq(api_key=api_key)

    for job in filtered_jobs:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Rate out of 5 the prestige of {job.company} using only 1 number nothing else",
                    }
                ],
                model="llama3-8b-8192",
            )

            prestige = chat_completion.choices[0].message.content.strip()
            if len(prestige) > 1 or not prestige.isdigit():
                prestige = "2"
            job.prestige = prestige
            final_job_list.append(job)
            job_data.append({
                'title': job.title,
                'link': job.link,
                'prestige': prestige,
            })
        except Exception as e:
            print(f"Error getting prestige for job {job.id}: {e}")

    return final_job_list, job_data


# Function to add jobs to Firestore with error handling
def add_jobs_to_firestore(jobs):
    for job in jobs:
        try:
            job.firestoreAdd()
        except Exception as e:
            print(f"Error adding job {job.id} to Firestore: {e}")


# Flask route to scrape jobs
@app.route('/server/scrape', methods=['POST'])
def get_jobs():
    data = request.json
    country_value = data.get('country', "")
    radius_value = data.get('radius', "")
    remote_value = data.get('remote', False)
    age_value = data.get('age', "")

    try:
        jobs = scrape_jobs_from_params(country_value, radius_value, remote_value, age_value)
    except Exception:
        return jsonify({"error": "Failed to scrape jobs."}), 500

    jobs_list = [Job(row['id'], row['title'], row['company'], row['description'], row['job_url'], 0, datetime.now()) for
                 _, row in jobs.iterrows() if row["job_type"] != "fulltime"]

    print(f"Found {len(jobs_list)} jobs.")

    filtered_jobs = filter_jobs(jobs_list)
    print(f"Number of filtered jobs: {len(filtered_jobs)}")

    final_job_list, job_data = get_job_prestige(filtered_jobs)
    add_jobs_to_firestore(final_job_list)

    return jsonify(job_data)

logging.basicConfig(level=logging.DEBUG)


# Flask route to get job recommendations
@app.route('/server/recommend', methods=['POST'])
def get_recommendations():
    try:
        uid = request.json.get('uid')

        # Initialize Firestore client
        db = firestore.client()

        # Fetch all jobs from Firestore
        jobs_ref = db.collection('jobs')
        job_data = [Jobs.from_firebase(doc.to_dict()) for doc in jobs_ref.stream()]
        for job in job_data:
            job.description = job.description or ''
        logging.info(f"Total jobs fetched: {len(job_data)}")

        # Helper function to get job IDs from user-specific subcollections
        def get_user_job_ids(collection_name):
            return {doc.id for doc in db.collection("user").document(uid).collection(collection_name).stream()}

        unliked_ids = get_user_job_ids("unliked")
        wishlisted_ids = get_user_job_ids("wishlisted")
        logging.info(f"Unliked job IDs: {unliked_ids}")
        logging.info(f"Wishlisted job IDs: {wishlisted_ids}")

        # Separate out unliked and wishlisted jobs from the job data
        unliked_jobs = [job for job in job_data if job.id in unliked_ids]
        wishlisted_jobs = [job for job in job_data if job.id in wishlisted_ids]

        # Filter out the unliked and wishlisted jobs from the main job data
        job_data = [job for job in job_data if job.id not in wishlisted_ids and job.id not in unliked_ids]
        logging.info(f"Jobs after filtering: {len(job_data)}")

        # Helper function to create a DataFrame from a list of jobs
        def create_job_dataframe(job_list):
            df = pd.DataFrame([{'id': job.id, 'description': job.description or '', 'title': job.title} for job in job_list])
            df['description'] = df['description'].fillna('')
            return df

        df_jobs = create_job_dataframe(job_data)
        df_unliked = create_job_dataframe(unliked_jobs)
        df_wishlist = create_job_dataframe(wishlisted_jobs)
        logging.info(f"DataFrames created: df_jobs={df_jobs.shape}, df_unliked={df_unliked.shape}, df_wishlist={df_wishlist.shape}")

        # Helper function to tokenize descriptions with TF-IDF vectorizer
        def tokenize_descriptions(df, vectorizer=None):
            if vectorizer is None:
                vectorizer = TfidfVectorizer(stop_words='english')
                vectors = vectorizer.fit_transform(df['description'])
            else:
                vectors = vectorizer.transform(df['description'])
            return vectors, vectorizer

        # Tokenize descriptions for all jobs, unliked jobs, and wishlisted jobs
        tfidf_matrix_jobs, vectorizer = tokenize_descriptions(df_jobs)
        tfidf_matrix_unliked, _ = tokenize_descriptions(df_unliked, vectorizer)
        tfidf_matrix_wishlist, _ = tokenize_descriptions(df_wishlist, vectorizer)

        # Compute cosine similarity matrices
        cosine_simJ_W = linear_kernel(tfidf_matrix_jobs, tfidf_matrix_wishlist)
        cosine_simJ_U = linear_kernel(tfidf_matrix_jobs, tfidf_matrix_unliked)

        # Step 2: Calculate mean similarity to wishlist (W)
        mean_sim_to_W = np.mean(cosine_simJ_W, axis=1)

        # Step 3: Calculate mean similarity to unliked (U)
        mean_sim_to_U = np.mean(cosine_simJ_U, axis=1)

        # Step 4: Calculate a combined score (higher is better for W, lower is better for U)
        combined_score = mean_sim_to_W - mean_sim_to_U

        # Step 5: Rank points based on the combined score
        ranked_indices = np.argsort(combined_score)[::-1]

        # Points in cosine_simJ that are closest to W and farthest from U
        top_points = ranked_indices[:10]  # Change 10 to however many top points you want to consider

        # Get the job IDs of the top points
        top_10_jobs = df_jobs.iloc[top_points]
        logging.info(f"Top 10 job IDs: {top_10_jobs['id'].tolist()}")

        # Return list of job IDs as JSON
        return jsonify(top_10_jobs['id'].tolist())

    except Exception as e:
        logging.error(f"Error in get_recommendations: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8000)
