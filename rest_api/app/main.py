from fastapi import FastAPI

import os

app = FastAPI()


@app.get("/")
def read_root():
    print_value = os.environ.get('HOSTNAME', 'DEFAULT_ENV')
    return {"Hello": f"From: {print_value}! I was changed again"}

@app.post("/api/reload-cache")
def reload_cache():
    #perform_reload_logic()
    return {"status": "Cache reloaded"}
