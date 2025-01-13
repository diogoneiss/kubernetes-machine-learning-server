import pathlib
import pickle

from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every

import os

app = FastAPI()

app.reload_counter = 0

import logging

logging.basicConfig(format="{levelname}:{name}:{message}", style="{")

#from pvc
BASE_DIR =  pathlib.Path(os.getenv("BASE_DIR", "datasets/"))

PICKLES_FOLDER = BASE_DIR / os.getenv("PICKLE_DIR", "pickles/")
K_BEST_TRACKS = int(os.getenv("K_BEST_TRACKS", "10"))
VERSION = os.getenv("VERSION", "V0.1")

RECOMMENDATIONS_FILE =os.getenv("RECOMMENDATIONS_FILE", "recommendations.pickle")
BEST_TRACKS_FILE = os.getenv("BEST_TRACKS_FILE", "best_tracks.pickle")
DATA_INVALIDATION_FILE = os.getenv("DATA_INVALIDATION_FILE", "last_execution.txt")

cache_file = BASE_DIR / DATA_INVALIDATION_FILE

def read_pickle_dict():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    PICKLES_FOLDER.mkdir(parents=True, exist_ok=True)

    # read best songs using path lib
    best_tracks_path = PICKLES_FOLDER / BEST_TRACKS_FILE
    with open(best_tracks_path, "rb") as f:
        best_tracks = pickle.load(f)

    logging.info(f"Best tracks loaded: {len(best_tracks)}")

    # read recommendations using path lib
    recommendations_path = PICKLES_FOLDER / RECOMMENDATIONS_FILE
    with open(recommendations_path, "rb") as f:
        recommendations = pickle.load(f)

    logging.info(f"Recommendations loaded: {len(recommendations)}")

    return best_tracks, recommendations

def is_data_stale():
    if not cache_file.exists():
        return True

    current_value = app.cache_value

    with open(cache_file, "r") as f:
        last_value = f.read()

    if current_value != last_value:
        app.cache_value = last_value
        return True

    return False

@app.on_event('startup')
@repeat_every(seconds=60)
def data_reload_handler():
    reload_data_if_required()

def reload_data_if_required():
    if not is_data_stale():
        logging.info("data is not stale, no need to reload")
        return

    logging.info("Reloading data!")
    app.best_tracks, app.recommendations = read_pickle_dict()

    app.reload_counter += 1
    logging.info(f"Finished reloading, should reflect changes. Data was reload {app.reload_counter} times")

@app.get("/")
def read_root():
    print_value = f"Last execution: {app.cache_value}. Total reloads: {app.reload_counter} and current version is {VERSION}"
    return {"Data": f"{print_value}"}

@app.post("/api/reload-data")
def reload_cache():
    reload_data_if_required()
    return {"status": "Data reloaded"}
