from clearml import Task, TaskTypes
from clearml.automation.controller import PipelineDecorator


@PipelineDecorator.component(
    return_values=['model'],
    task_type=TaskTypes.custom
)
def load_model():
    import pickle
    from clearml import Dataset
    data_path = Dataset.get(dataset_name="Models", alias="Models").get_local_copy()
    model_path = f"{data_path}/svd_model_full_data.sav"
    model = pickle.load(open(model_path, 'rb'))
    return model

@PipelineDecorator.component(
    return_values=['movie_ids'],
    task_type=TaskTypes.data_processing
)
def load_movies() -> list:
    from clearml import Dataset
    data_path = Dataset.get(dataset_name="Models", alias="Models").get_local_copy()
    movie_ids = set()
    with open(data_path + "/user_ratings.csv", 'r') as f:
        for line in f:
            movie_id = line.strip().split(',')[2]
            movie_ids.add(movie_id)
    movie_ids =  list(movie_ids)
    return movie_ids

@PipelineDecorator.component(
    return_values=['recs'],
    task_type=TaskTypes.inference,
    cache=True
)
def recommend_movies(user_id: int, movie_ids:list, model, top_k:int=10,) -> list:
    import random

    NEW_USER_RECS = ['the+shawshank+redemption+1994', 'once+upon+a+time+in+the+west+1968', 'schindlers+list+1993', 'life+is+beautiful+1997', 
                 'black+cat_+white+cat+1998', 'the+usual+suspects+1995', 'the+fall+2008', 'monsoon+wedding+2001', 'the+godfather+1972', 
                 'shadow+of+a+doubt+1943', 'the+lives+of+others+2006', 'all+quiet+on+the+western+front+1930', 'when+we+were+kings+1996', 
                 '3-iron+2004', 'aguirre+the+wrath+of+god+1972', 'the+celluloid+closet+1996', 'rashomon+1950', 'out+of+the+past+1947', 
                 'amlie+2001', 'stop+making+sense+1984']
    
    ratings = []
    for movie in movie_ids:
        prediction = model.predict(int(user_id), movie)
        ratings.append((prediction.iid, prediction.est))
    ratings.sort(key=lambda x: x[1], reverse=True)
    recs = [movie[0] for movie in ratings[:20]]
    if recs == NEW_USER_RECS:
        recs = random.sample([movie[0] for movie in ratings[:2*top_k]], top_k)
    return list(recs)

@PipelineDecorator.pipeline(name='My First Pipeline', project='My First Project', version='1.0.0')
def run_pipeline(dataset_name: str):
    model = load_model()
    movie_ids = load_movies()
    user_ids = [1, 2]
    for user_id in user_ids:
        recs = recommend_movies(user_id, movie_ids, model, 10)
        print(f"Recommendations for user {user_id}: {recs}")
    

if __name__ == '__main__':
    PipelineDecorator.run_locally()
    run_pipeline(dataset_name='Models')