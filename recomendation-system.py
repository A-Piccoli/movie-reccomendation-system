from flask import Flask, render_template,request, url_for
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from imdb import Cinemagoer
import requests

def download_movie_poster(movie_name):
    ia = Cinemagoer()
    movie_poster = ia.search_movie(movie_name)
    poster_response = requests.get(movie_poster[0]['cover url']) 
    if poster_response.status_code:
        fp = open(f"./static/{movie_name}.jpg", 'wb')
        fp.write(poster_response.content)
        fp.close()
        
def clear_poster_directory():
    path = "./static/"
    for file_name in os.listdir(path):
        # construct full file path
        file = path + file_name
        if os.path.isfile(file):
           os.remove(file)

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('home.html')

@app.route("/search",methods=["GET"])
def get_similiar():
    
    try:
    
        if len(os.listdir('./static/')) != 0:
            clear_poster_directory()
        else:
            pass
        
        movies_data = pd.read_csv("movie_dataset\movies.csv")

        selected_features = ['genres','keywords','tagline','cast','director']

        for feature in selected_features:
            movies_data[feature] = movies_data[feature].fillna('')
            

        combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

        vectorizer = TfidfVectorizer()

        feature_vectors = vectorizer.fit_transform(combined_features)

        similiarity = cosine_similarity(feature_vectors)
        # movie_name = input("Enter your favourite movie name: ")
        movie_name = request.args.get('movie_name')

        list_of_all_titles = movies_data['title'].tolist()

        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

        index_of_the_movie = movies_data[movies_data.title == find_close_match[0]]['index'].values[0]

        similarity_score = list(enumerate(similiarity[index_of_the_movie]))

        sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)

        print('Movies suggested for you : \n')
        movie_list = []
        i = 1
        if len(sorted_similar_movies) != 0:
            for movie in sorted_similar_movies:
                index = movie[0]
                title_from_index = movies_data[movies_data.index==index]['title'].values[0]
                if (i<=5):
                    try:
                        download_movie_poster(title_from_index)
                        movie_list.append(title_from_index)
                    except:
                        pass
                    i+=1
            return render_template('reccomend.html', movie_list=movie_list)
    
    except:
        return render_template('error.html')

    
        
if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)





