from fastapi import FastAPI

import os

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": f"From: {os.environ.get('HOSTNAME', 'DEFAULT_ENV')}"}

@app.post("/api/reload-cache")
def reload_cache():
    #perform_reload_logic()
    return {"status": "Cache reloaded"}
