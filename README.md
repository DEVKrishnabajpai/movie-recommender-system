# Movie Recommender System

A content-based movie recommendation system built using Machine Learning and Natural Language Processing.

## Features

- Recommends similar movies
- Uses genres, keywords, cast, crew and movie overview
- NLP preprocessing using stemming
- Cosine similarity based recommendations

## Dataset

TMDB 5000 Movies Dataset

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- NLTK

## Project Workflow

1. Data Cleaning
2. Feature Engineering
3. Text Preprocessing
4. Vectorization using CountVectorizer
5. Cosine Similarity Computation
6. Movie Recommendation

## How to Run

```bash
pip install -r requirements.txt
```

Open and run:

```bash
movie_recomendation_system.ipynb
```

## Sample Recommendation

Input:

Batman Begins

Output:

- The Dark Knight
- Batman
- Batman Returns
- The Dark Knight Rises
- Batman Forever
