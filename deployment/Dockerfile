# 
FROM python:3.10

# 
WORKDIR /code

# 
COPY ./requirement.txt /code/requirement.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirement.txt

# 
COPY ./app.py /code/app.py
COPY ./src /code/src
COPY ./models /code/models
# 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]