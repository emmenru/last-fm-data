# last-fm-data

## Table of Contents
1. [Project Overview](#project-overview)
2. [File Descriptions](#file-descriptions)
3. [Setup and Installation](#setup-and-installation)
4. [Important Notes](#important-notes)

## Project Overview
This repository contains code for scraping personal Last.fm music streaming data, building a SQL database, performing exploratory data analysis, and creating an AI-powered music discovery agent with natural language interface capabilities.

The system supports queries like "tell me about Radiohead", "my top artists last month", "find music like Radiohead" (artist information, listening habits, and music recommendations). 

### Background
Modern music streaming services are often focused on discovering 'new music' that you have not listened to before. However, quite often I find that looking back at what I have already listened to (and possibly forgotten about) could actually lead me into more interesting directions for music discovery. 

### Aim
To create a complete data science pipeline that would allow me to explore my listening habits through interaction using natural language (as opposed to defining multiple SQL queries to do so). 

### Approach
- **Data Collection**: Automated scraping of Last.fm data using the official API
- **Database Design**: SQL schema design and database creation 
- **Exploratory Analysis**: Interactive visualizations of listening patterns
- **AI Agent Development**: Workflow orchestration of simple AI chat interface based on sentence transformer models 

### Findings
Please refer to step **4. View Results** for interactive exploration of results. 

## File Descriptions

### Config
  - **schema.sql**: SQL database schema definitions
  - **config.py**: Configuration for Last.fm data collection

### Data
  - **lastfm_data.db**: SQLite database (music data snapshot collected 2025-05-08)
  - **collection.log**: Log file

### Figs 
- Figures (db schema, node graph)

### Utils
  **Data Collection & Database:**
  - **data_collector.py**: Main data collection orchestration
  - **database_helper.py**: Database operations and SQL query management
  - **lastfm_api.py**: Last.fm API interface wrapper 
  
  **AI Agent:**
  - **music_discovery_helper.py**: Core AI agent implementation featuring:
    - **SentenceTransformer Integration**: `all-MiniLM-L6-v2` model for semantic understanding (intent classification and entity extraction)
    - **LangGraph Workflow**: 7-node processing pipeline 

### Notebooks
  - **EDA_notebook.ipynb**: EDA and interactive Plotly visualizations
  - **lastfm_notebook.ipynb**: Workflow for data collection and database setup
  - **music_discovery_notebook.ipynb**: AI-powered music discovery agent with simple chat interface

### Database Schema

ERDiagram available here: https://dbdiagram.io/d/last-fm_erdiagram-68a9d7b11e7a6119674638b6

## Setup and Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/emmenru/last-fm-data.git
    cd last-fm-data
    ```

2. **Environment Setup**:
   - In Google Colab: grant access to drive, make sure paths are set up correctly, and set up the following secrets:
      - `LASTFM_API_KEY`: Your Last.fm API key
      - `USERNAME`: Your Last.fm username
   - Access secrets in notebooks using:
    ```python
    from google.colab import userdata
    api_key = userdata.get('LASTFM_API_KEY')
    username = userdata.get('USERNAME')
    ```

3. **Run the Analysis**:
    - Execute notebooks in order:
        1. `lastfm_notebook.ipynb`
        2. `EDA_notebook.ipynb` 
        3. `music_discovery_notebook.ipynb` 

4. **View Results**:
    - **Data Collection**: 
      [![View on NBViewer](https://img.shields.io/badge/Data%20Collection-NBViewer-orange?style=for-the-badge&logo=jupyter)](https://nbviewer.org/github/emmenru/last-fm-data/blob/main/lastfm_notebook.ipynb)
      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emmenru/last-fm-data/blob/main/lastfm_notebook.ipynb)
    
    - **EDA Analysis**: 
      [![View on NBViewer](https://img.shields.io/badge/EDA%20Analysis-NBViewer-orange?style=for-the-badge&logo=jupyter)](https://nbviewer.org/github/emmenru/last-fm-data/blob/main/EDA_notebook.ipynb)
      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emmenru/last-fm-data/blob/main/EDA_notebook.ipynb)
    
    - **AI Music Discovery**: 
      [![View Music Discovery on NBViewer](https://img.shields.io/badge/Music%20Discovery-NBViewer-orange?style=for-the-badge&logo=jupyter)](https://nbviewer.org/github/emmenru/last-fm-data/blob/main/music_discovery_notebook.ipynb)
      [![Open Music Discovery In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emmenru/last-fm-data/blob/main/music_discovery_notebook.ipynb)
      
## Important Notes
- Data collected on 2025-05-08 using Last.fm API (only a snapshot of personal listening history). 
- Note that all timestamps are in UTC if nothing else is stated (see details in EDA, where labels 'paris' are added when timezone conversion is used - this is done only for table listening_df). 
- Tracks do not have full album coverage (about 35%). Some tracks do not have detailed meta data (resulting e.g. in duration of 0).
- Additional API requests are sent to find meta data for artists tags. This means that if an artist has 10 tags 10 calls will be sent.
- Backup is set to Google drive since Colab is used.
- Similarity Data: Tables: artist_similar: finds 10 similar artists for resp. artist, and stores this data if exists in the database before creating relationship.
