# Semantic Movie Search Engine

A content-based movie recommendation system that uses sentence transformers to convert movie descriptions and queries into embeddings, then finds the most semantically similar matches.

## Features

1. Semantic similarity matching using sentence transformers
2. WebUI built with Streamlit
3. Command line interface

## Implementation Details

1. The search engine uses the `nomic-ai/nomic-embed-text-v1.5` model to create embeddings for both the movie descriptions and search queries. 
2. It uses cosine similarity to find the most relevant matches.

### Installation

1. Clone the repository:
   
   ```bash
   git clone https://github.com/Kira-Pgr/lumaa-spring-2025-ai-ml.git
   cd lumaa-spring-2025-ai-ml
   ```

2. Install the required dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```

### Dataset

The system uses a movie dataset (`dataset.csv`) with 500 rows that contains two main columns:

- `Title`: The name of the movie
- `PlotSummary`: A text description of the movie's plot

Make sure the `dataset.csv` file is in the root directory of the project.

### Usage

#### Option 1: Web Interface

1. Start the Streamlit web interface:
   
   ```bash
   streamlit run webui.py
   ```
   
   If ran successfully, it will automatically open a new browser window displaying webui.

2. Enter your search query and adjust the number of results as needed

3. 

#### Option 2: CLI

To use the command line interface:

```bash
python search_engine.py -q "movies about love" -k 5
```

This will run a sample search with the query "movies about love". 

##### Example Output

```
Search Results:
Title:  Angst, Summary: Angst tells the story of a group of horror film devotees living in Sydney's King's Cross. It has been described as an Australian version of Kevin Smith's Clerks and Mallrats with the same Dark humor as SubUrbia. The film stars Sam Lewis as a cynical, sexually frustrated video store employee with a bad case of unresolved love., Similarity: 0.627
Title: Dolls, Summary: The film is not in strict chronological order, but there is a strong visual emphasis on the changing of the seasons and the bonds of love over the progression of time. The film leads into it by opening with a performance of Bunraku theatre, and closes with a shot of dolls from the same., Similarity: 0.626
```

### Video Demo

![Demo](videos/demo.gif)

### Project Structure

- `search_engine.py`: Core search functionality and CLI interface
- `webui.py`: Streamlit web interface
- `dataset.csv`: Movie dataset
- `requirements.txt`: Python dependencies
- `videos/`: Directory for video demonstrations
- `README.md`: README file
