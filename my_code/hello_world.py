from clearml import Task
import pickle, os
import random

task = Task.init(project_name='My First Project', task_name='My First Task', output_uri=True)

MODEL_PATH = 'models/svd_model_full_data.sav'
MOVIE_IDS_PATH = 'models/user_ratings.csv'

params = {
    'K': 20,
}

model = pickle.load(open(MODEL_PATH, 'rb'))
pickle.dump(model, open('model.sav', 'wb'))

task.upload_artifact(
    name='model.sav', 
    artifact_object=os.path.join(
        'model.sav'
    )
)

movie_ids = set()
with open(MOVIE_IDS_PATH, 'r') as f:
    for line in f:
        movie_id = line.strip().split(',')[2]
        movie_ids.add(movie_id)
movie_ids =  list(movie_ids)

def recommend_movies(user_id: int, top_k:int=params['K']) -> list:
    ratings = []
    for movie in movie_ids:
        prediction = model.predict(int(user_id), movie)
        ratings.append((prediction.iid, prediction.est))
    ratings.sort(key=lambda x: x[1], reverse=True)
    recs = [movie[0] for movie in ratings[:20]]
    if recs == NEW_USER_RECS:
        recs = random.sample([movie[0] for movie in ratings[:40]], 20)
    return list(recs)

task.connect(params)
