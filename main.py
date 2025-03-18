import os
import numpy as np
import pandas as pd
from firebase_admin import firestore
from flask import Flask, jsonify, request
import logging
from groq import Groq
from jobspy import scrape_jobs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from py.util import config
from py.util.firebase import initialize_firebase
from py.util.jobs import Job, Jobs
from datetime import datetime
from flask_cors import CORS

# Initialize Flask app and configurations
app = Flask(__name__)
CORS(app)
repository_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
os.chdir(repository_root)
initialize_firebase(config.FIREBASE_CREDENTIALS)
logging.basicConfig(level=logging.DEBUG)


def filter_jobs(jobs_list):
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
        if (all(keyword not in description_lower for keyword in exclude_keywords) and
                any(keyword in description_lower for keyword in include_keywords) and
                any(phrase in description_lower for phrase in include_phrases) and
                all(phrase not in description_lower for phrase in exclude_phrases) and
                all(title not in title_lower for title in exclude_titles)):
            filtered_jobs.append(job)

    return filtered_jobs


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
        return jobs
    except Exception as e:
        logging.error(f"Error scraping jobs: {e}")
        raise


def get_job_prestige(filtered_jobs):
    final_job_list = []
    job_data = []
    client = Groq(api_key=config.GROQ_API_KEY)

    for job in filtered_jobs:
        try:
            chat_completion = client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": f"Rate out of 5 the prestige of {job.company} using only 1 number nothing else"
                }],
                model="llama3-8b-8192",
            )
            prestige = chat_completion.choices[0].message.content.strip()
            prestige = "2" if len(prestige) > 1 or not prestige.isdigit() else prestige
            job.prestige = prestige
            final_job_list.append(job)
            job_data.append({'title': job.title, 'link': job.link, 'prestige': prestige})
        except Exception as e:
            logging.error(f"Error getting prestige for job {job.id}: {e}")

    return final_job_list, job_data


def add_jobs_to_firestore(jobs):
    for job in jobs:
        try:
            job.firestoreAdd()
        except Exception as e:
            logging.error(f"Error adding job {job.id} to Firestore: {e}")


@app.route('/', methods=['POST'])
def startpoint():
    return jsonify("Success")


@app.route('/server/scrape', methods=['POST'])
def get_jobs():
    data = request.json
    country = data.get('country', "")
    radius = data.get('radius', "")
    remote = data.get('remote', False)
    age = data.get('age', "")

    try:
        jobs = scrape_jobs_from_params(country, radius, remote, age)
    except Exception:
        return jsonify({"error": "Failed to scrape jobs."}), 500

    jobs_list = [Job(row['id'], row['title'], row['company'], row['description'], row['job_url'], 0, datetime.now())
                 for _, row in jobs.iterrows() if row["job_type"] != "fulltime"]

    filtered_jobs = filter_jobs(jobs_list)
    final_job_list, job_data = get_job_prestige(filtered_jobs)
    add_jobs_to_firestore(final_job_list)

    return jsonify(job_data)


@app.route('/server/recommend', methods=['POST'])
def get_recommendations():
    try:
        # Step 1: Get user ID from the request
        print("Getting user ID from request")
        uid = request.json.get('uid')

        # Step 2: Initialize Firestore client
        print("Initializing Firestore client")
        db = firestore.client()

        # Step 3: Retrieve all job data from Firestore
        print("Retrieving job data from Firestore")
        jobs_ref = db.collection('jobs')
        job_data = [Jobs.from_firebase(doc.to_dict()) for doc in jobs_ref.stream()]

        # Step 4: Define a helper function to get user-specific job IDs
        def get_user_job_ids(collection_name):
            print(f"Retrieving job IDs for collection: {collection_name}")
            collection_ref = db.collection("user").document(uid).collection(collection_name)
            docs = collection_ref.stream()
            return {doc.id for doc in docs}

        # Step 5: Get unliked and wishlisted job IDs
        print("Getting unliked and wishlisted job IDs")
        unliked_ids = get_user_job_ids("unliked") if db.collection("user").document(uid).collection(
            "unliked").stream() else set()
        wishlisted_ids = get_user_job_ids("wishlisted") if db.collection("user").document(uid).collection(
            "wishlisted").stream() else set()

        # Step 6: Separate jobs into unliked, wishlisted, and neutral categories
        print("Separating jobs into unliked, wishlisted, and neutral categories")
        unliked_jobs = [job for job in job_data if job.id in unliked_ids]
        wishlisted_jobs = [job for job in job_data if job.id in wishlisted_ids]
        job_data = [job for job in job_data if job.id not in wishlisted_ids and job.id not in unliked_ids]

        # Step 7: Define a function to create a DataFrame from job list
        def create_job_dataframe(job_list):
            print("Creating DataFrame for job list")
            df = pd.DataFrame(
                [{'id': job.id, 'description': job.description if job.description else '', 'title': job.title} for job
                 in job_list])
            df['description'] = df['description'].fillna('')
            print(df.head())  # Debugging: Print the first few rows of the DataFrame
            return df

        # Step 8: Create DataFrames for different job categories
        print("Creating DataFrames for jobs, unliked jobs, and wishlisted jobs")
        df_jobs = create_job_dataframe(job_data)
        df_unliked = create_job_dataframe(unliked_jobs)
        df_wishlist = create_job_dataframe(wishlisted_jobs)

        # Step 9: Define a function to tokenize descriptions and create TF-IDF matrix
        def tokenize_descriptions(df, vectorizer=None):
            print("Tokenizing descriptions")
            if 'description' not in df.columns:
                raise ValueError("DataFrame must contain a 'description' column")
            if vectorizer is None:
                vectorizer = TfidfVectorizer(stop_words='english')
                vectors = vectorizer.fit_transform(df['description'])
            else:
                vectors = vectorizer.transform(df['description'])
            return vectors, vectorizer

        # Step 10: Tokenize job descriptions and create TF-IDF matrices
        print("Creating TF-IDF matrices for job descriptions")
        tfidf_matrix_jobs, vectorizer = tokenize_descriptions(df_jobs)
        tfidf_matrix_unliked, _ = tokenize_descriptions(df_unliked, vectorizer)
        tfidf_matrix_wishlist, _ = tokenize_descriptions(df_wishlist, vectorizer)

        # Step 11: Compute cosine similarities
        print("Computing cosine similarities")
        cosine_simJ_W = linear_kernel(tfidf_matrix_jobs, tfidf_matrix_wishlist)
        cosine_simJ_U = linear_kernel(tfidf_matrix_jobs, tfidf_matrix_unliked)

        # Step 12: Calculate mean similarity scores
        print("Calculating mean similarity scores")
        mean_sim_to_W = np.mean(cosine_simJ_W, axis=1)
        mean_sim_to_U = np.mean(cosine_simJ_U, axis=1)

        # Step 13: Combine scores and rank jobs
        print("Combining scores and ranking jobs")
        combined_score = mean_sim_to_W - mean_sim_to_U
        ranked_indices = np.argsort(combined_score)[::-1]
        top_points = ranked_indices[:10]

        # Step 14: Get top 10 recommended jobs
        print("Retrieving top 10 recommended jobs")
        top_10_jobs = df_jobs.iloc[top_points]

        # Step 15: Return the IDs of the top 10 jobs
        print("Returning top 10 job IDs")
        return jsonify(top_10_jobs['id'].tolist())

    except Exception as e:
        # Step 16: Log and return the error if an exception occurs
        logging.error(f"Error in get_recommendations: {str(e)}")
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
