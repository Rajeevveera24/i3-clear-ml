from clearml import Task, Dataset
import pickle, os
import random

task = Task.init(project_name='My First Project', task_name='Dataset Task', output_uri=True)

NEW_USER_RECS = ['the+shawshank+redemption+1994', 'once+upon+a+time+in+the+west+1968', 'schindlers+list+1993', 'life+is+beautiful+1997', 
                 'black+cat_+white+cat+1998', 'the+usual+suspects+1995', 'the+fall+2008', 'monsoon+wedding+2001', 'the+godfather+1972', 
                 'shadow+of+a+doubt+1943', 'the+lives+of+others+2006', 'all+quiet+on+the+western+front+1930', 'when+we+were+kings+1996', 
                 '3-iron+2004', 'aguirre+the+wrath+of+god+1972', 'the+celluloid+closet+1996', 'rashomon+1950', 'out+of+the+past+1947', 
                 'amlie+2001', 'stop+making+sense+1984']

MOVIE_IDS_PATH = 'models/user_ratings.csv'

params = {'K': 20}
task.connect(params)

data_path = Dataset.get(dataset_name="Models", alias="Models").get_local_copy()
print(data_path)

model_path = f"{data_path}/svd_model_full_data.sav"
model = pickle.load(open(model_path, 'rb'))

movie_ids = set()
with open(data_path + "/user_ratings.csv", 'r') as f:
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
