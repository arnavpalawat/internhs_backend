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
        uid = request.json.get('uid')
        db = firestore.client()

        jobs_ref = db.collection('jobs')
        job_data = [Jobs.from_firebase(doc.to_dict()) for doc in jobs_ref.stream()]
        for job in job_data:
            job.description = job.description or ''

        def get_user_job_ids(collection_name):
            return {doc.id for doc in db.collection("user").document(uid).collection(collection_name).stream()}

        unliked_ids = get_user_job_ids("unliked")
        wishlisted_ids = get_user_job_ids("wishlisted")

        unliked_jobs = [job for job in job_data if job.id in unliked_ids]
        wishlisted_jobs = [job for job in job_data if job.id in wishlisted_ids]
        job_data = [job for job in job_data if job.id not in wishlisted_ids and job.id not in unliked_ids]

        def create_job_dataframe(job_list):
            df = pd.DataFrame([{'id': job.id, 'description': job.description or '', 'title': job.title} for job in job_list])
            df['description'] = df['description'].fillna('')
            return df

        df_jobs = create_job_dataframe(job_data)
        df_unliked = create_job_dataframe(unliked_jobs)
        df_wishlist = create_job_dataframe(wishlisted_jobs)

        def tokenize_descriptions(df, vectorizer=None):
            if vectorizer is None:
                vectorizer = TfidfVectorizer(stop_words='english')
                vectors = vectorizer.fit_transform(df['description'])
            else:
                vectors = vectorizer.transform(df['description'])
            return vectors, vectorizer

        tfidf_matrix_jobs, vectorizer = tokenize_descriptions(df_jobs)
        tfidf_matrix_unliked, _ = tokenize_descriptions(df_unliked, vectorizer)
        tfidf_matrix_wishlist, _ = tokenize_descriptions(df_wishlist, vectorizer)

        cosine_simJ_W = linear_kernel(tfidf_matrix_jobs, tfidf_matrix_wishlist)
        cosine_simJ_U = linear_kernel(tfidf_matrix_jobs, tfidf_matrix_unliked)

        mean_sim_to_W = np.mean(cosine_simJ_W, axis=1)
        mean_sim_to_U = np.mean(cosine_simJ_U, axis=1)
        combined_score = mean_sim_to_W - mean_sim_to_U
        ranked_indices = np.argsort(combined_score)[::-1]
        top_points = ranked_indices[:10]

        top_10_jobs = df_jobs.iloc[top_points]

        return jsonify(top_10_jobs['id'].tolist())

    except Exception as e:
        logging.error(f"Error in get_recommendations: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
