#FROM python:3.9
#
#WORKDIR /code
#
#COPY ./requirements.txt /code/requirements.txt
#
#RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
#
#COPY ./app /code/app
#
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

# src: https://docs.astral.sh/uv/guides/integration/fastapi/#migrating-an-existing-fastapi-project

FROM python:3.12-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
COPY . /app

# Install the application dependencies.
WORKDIR /app
RUN uv sync --frozen --no-cache

# Run the application.
CMD ["./.venv/bin/fastapi", "run", "app/main.py", "--port", "80", "--host", "0.0.0.0"]