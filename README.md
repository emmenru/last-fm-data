# last-fm-data
 Digging into scrobbled music data to find recommendations

## 📊 View the Analysis

[![View on NBViewer](https://img.shields.io/badge/View%20on-NBViewer-orange?style=for-the-badge&logo=jupyter)](https://nbviewer.org/github/emmenru/last-fm-data/blob/main/EDA.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emmenru/last-fm-data/blob/main/EDA.ipynb)

- **NBViewer**: View all visualizations, outputs, and analysis results (read-only)
- **Colab**: Interactive version where you can run and modify the code

## 🎵 What's Inside

- **Data Collection**: ETL pipeline using Last.fm API
- **Exploratory Analysis**: Listening patterns, time-based insights, network analysis, interative visualizations (plotly)

## 🔧 Technical Notes
 - Note that all timestamps are in UTC if nothing else is stated (see details in EDA, where labels 'paris' are added when timezone conversion is used - this is done only for table listening_df)
 - Tracks do not have full album coverage, seems to be at 35% currently. 
 Some tracks do not have detailed meta data (resulting e.g. in duration of 0). 
 - Additional API requests are sent to find meta data for artists tags. This means that if an artist has 10 tags 10 calls will be sent. 
-  Backup is set to Google drive since Colab is used. 
- Tables: artist_similar: finds 10 similar artists for resp. artist, and stores this data if exists in the database before creating relationship 