# webui.py
# A webui for the search engine
import streamlit as st
import pandas as pd
from search_engine import SentenceTransformerSearchEngine
import torch

torch.classes.__path__ = []

st.set_page_config(
    page_title="Movie Search Engine",
)

st.title("Semantic Movie Search Engine")
st.markdown(
    """On first run, the search engine will take a while to load and download the embedding model. Please be patient.""")

@st.cache_resource
def index_documents():
    """Load data from dataset.csv"""
    engine = SentenceTransformerSearchEngine()
    engine.load_data('dataset.csv')
    return engine

try:
    search_engine = index_documents()
    with st.form("search_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Query:",
                placeholder="E.g., movies about love"
            )
        
        with col2:
            num_results = st.number_input(
                "Number of results",
                min_value=1,
                max_value=20,
                value=5
            )
            
        submit = st.form_submit_button("Search")
        
    if submit and query:
        results_df, sim_scores = search_engine.search(query, top_k=num_results)
        for idx, score in zip(results_df.index, sim_scores):
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"### {results_df.loc[idx, 'Title']}")
                    st.markdown(results_df.loc[idx, 'PlotSummary'])
                
                with col2:
                    st.metric(
                        label="Relevance Score",
                        value=f"{score:.2%}"
                    )
                
                st.divider()

except Exception as e:
    st.error(f"""
        Error loading the search engine.
        Error {str(e)}
    """)
