from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/search")
def get_similiar():
    movies_data = pd.read_csv('movie_dataset\movies.csv')

selected_features = ['genres','keywords','tagline','cast','director']

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
    

combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)

similiarity = cosine_similarity(feature_vectors)

movie_name = input("Enter your favourite movie name: ")

list_of_all_titles = movies_data['title'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

print(f"The closest match for {movie_name} is: {find_close_match[0]}")

index_of_the_movie = movies_data[movies_data.title == find_close_match[0]]['index'].values[0]

similarity_score = list(enumerate(similiarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)

print('Movies suggested for you : \n')
suggestion_list = []
i = 1
for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1