import asyncio
import pathlib
import pickle
import random
import sys
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi_utils.tasks import repeat_every
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] [%(levelname)s] %(name)s: %(message)s")

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(consoleHandler)

logger.info('API is starting up')

load_dotenv()

BASE_DIR =  pathlib.Path(os.getenv("BASE_DIR", "machine-learning/api-data/"))
PICKLES_FOLDER = BASE_DIR / os.getenv("PICKLE_DIR", "pickles/")
K_BEST_TRACKS = int(os.getenv("K_BEST_TRACKS", "10"))
VERSION = os.getenv("VERSION", "V0.1")
POLLING_WAIT_IN_MINUTES = int(os.getenv("POLLING_WAIT_IN_MINUTES", "1"))
RECOMMENDATIONS_FILE =os.getenv("RECOMMENDATIONS_FILE", "recommendations.pickle")
BEST_TRACKS_FILE = os.getenv("BEST_TRACKS_FILE", "best_tracks.pickle")
DATA_INVALIDATION_FILE = os.getenv("DATA_INVALIDATION_FILE", "last_execution.txt")

cache_file = BASE_DIR / DATA_INVALIDATION_FILE

APP_PATH_FROM_ROOT = pathlib.Path(os.getenv("APP_PATH_FROM_ROOT", "rest_api/"))

TEMPLATES_DIR = APP_PATH_FROM_ROOT / "templates/"
STATIC_DIR = APP_PATH_FROM_ROOT / "static/"

logger.info(f"Will search for templates on {TEMPLATES_DIR}")
logger.info(f"Contents from current execution directory: {os.listdir()} and contents from APP_PATH_FROM_ROOT: {os.listdir(APP_PATH_FROM_ROOT)}")

def read_pickle_dict():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    PICKLES_FOLDER.mkdir(parents=True, exist_ok=True)

    best_tracks_path = PICKLES_FOLDER / BEST_TRACKS_FILE

    # print("Best tracks path is ", best_tracks_path)
    #
    # print("Files in current directory: ", os.listdir())
    #
    # print("Files in base dir: ", os.listdir(BASE_DIR))

    if not best_tracks_path.exists():
        logger.error(f"Best tracks file not found at {best_tracks_path}")
        raise ValueError(f"Best tracks file not found at {best_tracks_path}")

    with open(best_tracks_path, "rb") as f:
        best_tracks = pickle.load(f)

    logger.info(f"Best tracks loaded: {len(best_tracks)}")

    # read recommendations using path lib
    recommendations_path = PICKLES_FOLDER / RECOMMENDATIONS_FILE
    with open(recommendations_path, "rb") as f:
        recommendations = pickle.load(f)

    logger.info(f"Recommendations loaded: {len(recommendations)}")

    return best_tracks, recommendations

def is_data_stale():
    if not cache_file.exists():
        logger.info("Cache file does not exist")
        return True

    current_value = app.cache_value

    with open(cache_file, "r") as f:
        last_value = f.read()

    if current_value != last_value:
        logger.info(f"Data is stale, current value is {current_value} and last value was {last_value}")
        app.cache_value = last_value
        return True

    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    await data_reload_handler()
    yield
    logger.info("Exiting...")

@repeat_every(seconds=60*POLLING_WAIT_IN_MINUTES)
def data_reload_handler():
    reload_data_if_required()

def reload_data_if_required():
    if not is_data_stale() and app.finished_loading:
        logger.info("data is not stale, no need to reload")
        return
    perform_data_reload()

def perform_data_reload():
    logger.info("Reloading data!")
    app.best_tracks, app.recommendations = read_pickle_dict()

    app.reload_counter += 1
    logger.info(f"Finished reloading, should reflect changes. Data was reloaded {app.reload_counter} times")
    app.finished_loading = True

tags_metadata = [
    {
        "name": "recommend",
        "description": "Song recommendation service",
    },
]
app = FastAPI(
    title="Music Recommendation API",
    version=VERSION,
    summary="Kubernetes based deployment with fpgrowth recommendations",
    openapi_tags=tags_metadata,
    lifespan=lifespan

)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


templates = Jinja2Templates(directory=TEMPLATES_DIR)

app.reload_counter = 0
app.cache_value = None
app.finished_loading = False
app.recommendations = None
app.best_tracks = None
app.sample_songs = None

@app.get("/test", tags=["util"], include_in_schema=False)
def redirect_to_doc():
    response = RedirectResponse(url='/docs#/recommend/get_recommendations_api_recommend__post')
    return response

class SongRequest(BaseModel):
    songs: List[str]

openApiBody = Body(
            openapi_examples={
                "normal": {
                    "summary": "Common songs",
                    "description": "Will give normal recommendations",
                    "value":  {"songs": ["Gold Digger", "Closer"]},
                },
                "uncommon": {
                    "summary": "Songs not that common",
                    "value": {"songs": ["The Motto", "Despacito"]},
                },
                "absent": {
                    "summary": "Songs without recommendations",
                    "value":  {"songs": ["Evidencias", "Esse cara sou eu"]},
                },
            },
        )

@app.post("/api/recommend/", tags=["recommend"])
def get_recommendations(request: Annotated[SongRequest, openApiBody]):
    if not request.songs:
        raise HTTPException(status_code=400, detail="The songs list cannot be empty.")

    recommended_songs = recommend_tracks_for_track(request.songs)

    return {
        "songs": recommended_songs,
        "model_date": app.cache_value,
        "version": VERSION
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def render_client(request: Request):
    if app.best_tracks is None:
        logger.info("Best tracks not loaded, waiting for data to be loaded")
        await asyncio.sleep(2)

    random_song = random.choice(app.best_tracks)
    seed_tracks = [random_song["track_name"]]

    tracks = perform_static_recommendation(seed_tracks)

    return templates.TemplateResponse(
        request=request, name="client.html", context={"tracks": tracks}
    )

def perform_static_recommendation(seed_tracks: list[str]):
    # tracks in deterministic order
    sorted_tracks = sorted(seed_tracks)

    seed = hash(tuple(sorted_tracks))

    # Seed the random number generator
    random.seed(seed)
    best_tracks = app.best_tracks

    if not best_tracks:
        logger.error("Best tracks not loaded")
        return ["No recommendations available at the moment"]

    recommendations = random.sample(best_tracks, k=K_BEST_TRACKS)

    songs = [x["track_name"] for x in recommendations]
    return songs

def recommend_tracks_for_track(seed_tracks: list[str]):
    if not app.recommendations:
        logger.error("Recommendations not loaded, calling pickle reload")
        reload_data_if_required()
        return ["No recommendations available at the moment"]
    songs_to_song_sets = app.recommendations

    if not songs_to_song_sets:
        logger.error("Recommendations not loaded")
        return ["No recommendations available at the moment"]

    songs_present = [seed_track for seed_track in seed_tracks if seed_track in songs_to_song_sets]
    if not songs_present:
        logger.warning(f"Tracks [{seed_tracks}] not found in the song recommendation list.")
        return perform_static_recommendation(seed_tracks)

    merged_recommendations = defaultdict(int)

    for song in songs_present:
        new_songs = songs_to_song_sets[song]
        for recommentation in new_songs:
            value = new_songs[recommentation]
            # Keep the maximum value for each track
            merged_recommendations[recommentation] = max(merged_recommendations[recommentation], value)

    # sort dict by value
    sorted_by_confidence = sorted(merged_recommendations.items(), key=lambda x: x[1], reverse=True)

    recommended = sorted_by_confidence[:K_BEST_TRACKS]
    song_names = [song[0] for song in recommended]
    return song_names