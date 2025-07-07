# utils/music_discovery_helper.py
"""
Helper classes and functions for the Music Discovery AI Agent
"""

import os
import sqlite3
import requests
import json
import re
from typing import TypedDict, List, Dict, Any, Tuple

# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """State that flows through the agent nodes"""
    user_query: str
    intent: str
    confidence: float
    entities: List[str]
    sql_query: str
    query_results: List[Dict[str, Any]]
    api_results: List[Dict[str, Any]]
    response: str
    error_message: str
    needs_api_call: bool

# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    """Handles all database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute SQL query and return results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"❌ Database error: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test database connection and return success status"""
        try:
            tables = self.execute_query("SELECT name FROM sqlite_master WHERE type='table'")
            print(f"✅ Database: {len(tables)} tables")
            return len(tables) > 0
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def get_artist_names(self, limit: int = 200) -> List[str]:
        """Get list of artist names for entity extraction"""
        try:
            query = "SELECT DISTINCT name FROM artists ORDER BY listeners DESC LIMIT ?"
            results = self.execute_query(query, (limit,))
            artist_names = [row['name'].lower() for row in results]
            print(f"📊 Loaded {len(artist_names)} artist names")
            return artist_names
        except Exception as e:
            print(f"⚠️ Could not load artist names: {e}")
            return []

# =============================================================================
# HUGGING FACE HELPER
# =============================================================================

class HuggingFaceHelper:
    """Helper class for Hugging Face API calls"""
    
    def __init__(self, hf_token: str = None):
        self.hf_token = hf_token
        self.headers = {}
        if hf_token:
            self.headers["Authorization"] = f"Bearer {hf_token}"
        
        # API endpoints
        self.classification_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
        
        # Intent labels for classification
        self.intent_labels = [
            "listening analysis and user statistics",
            "music recommendations and suggestions", 
            "genre and tag exploration",
            "artist information and biography"
        ]
    
    def classify_intent(self, query: str) -> Tuple[str, float]:
        """Classify intent using HF zero-shot classification"""
        payload = {
            "inputs": query,
            "parameters": {
                "candidate_labels": self.intent_labels
            }
        }
        
        try:
            response = requests.post(
                self.classification_url, 
                headers=self.headers, 
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle potential error responses
                if isinstance(result, dict) and 'error' in result:
                    print(f"HF API error: {result['error']}")
                    return self._fallback_classification(query)
                
                best_label = result['labels'][0]
                confidence = result['scores'][0]
                
                # Map to our intent categories
                intent_mapping = {
                    "listening analysis and user statistics": "LISTENING_ANALYSIS",
                    "music recommendations and suggestions": "RECOMMENDATIONS",
                    "genre and tag exploration": "GENRE_EXPLORATION", 
                    "artist information and biography": "ARTIST_INFO"
                }
                
                intent = intent_mapping.get(best_label, "LISTENING_ANALYSIS")
                return intent, confidence
                
            else:
                print(f"HF API returned {response.status_code}")
                return self._fallback_classification(query)
                
        except Exception as e:
            print(f"HF classification error: {e}")
            return self._fallback_classification(query)
    
    def _fallback_classification(self, query: str) -> Tuple[str, float]:
        """Rule-based fallback if HF API fails"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['top', 'best', 'most', 'stats', 'listening']):
            return 'LISTENING_ANALYSIS', 0.8
        elif any(word in query_lower for word in ['recommend', 'similar', 'like', 'find']):
            return 'RECOMMENDATIONS', 0.8
        elif any(word in query_lower for word in ['genre', 'tag', 'style', 'explore']):
            return 'GENRE_EXPLORATION', 0.8
        elif any(word in query_lower for word in ['about', 'tell me', 'who is', 'info']):
            return 'ARTIST_INFO', 0.8
        else:
            return 'LISTENING_ANALYSIS', 0.6

# =============================================================================
# LAST.FM API HELPER
# =============================================================================

class LastFMAPIHelper:
    """Handles Last.fm API calls for extended recommendations"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://ws.audioscrobbler.com/2.0/"
    
    def get_similar_artists(self, artist_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get similar artists from Last.fm API"""
        if not self.api_key:
            return []
            
        params = {
            'method': 'artist.getsimilar',
            'artist': artist_name,
            'api_key': self.api_key,
            'format': 'json',
            'limit': limit
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if 'similarartists' in data and 'artist' in data['similarartists']:
                return data['similarartists']['artist']
            return []
        except Exception as e:
            print(f"Last.fm API error: {e}")
            return []
    
    def get_artist_info(self, artist_name: str) -> Dict[str, Any]:
        """Get detailed artist information from Last.fm API"""
        if not self.api_key:
            return {}
            
        params = {
            'method': 'artist.getinfo',
            'artist': artist_name,
            'api_key': self.api_key,
            'format': 'json'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            return data.get('artist', {})
        except Exception as e:
            print(f"Last.fm API error: {e}")
            return {}

# =============================================================================
# ENTITY EXTRACTION
# =============================================================================

def extract_entities(query: str, artist_names: List[str]) -> List[str]:
    """Extract entities from user query"""
    entities = []
    query_lower = query.lower()
    
    # Extract numbers
    numbers = re.findall(r'\b(\d+)\b', query_lower)
    entities.extend(numbers)
    
    # Extract artist names (check against database)
    for artist_name in artist_names:
        if artist_name in query_lower and len(artist_name) > 3:
            entities.append(artist_name)
            break  # Take first match
    
    # Extract common genres
    genres = ['indie', 'rock', 'pop', 'jazz', 'electronic', 'classical', 
             'hip hop', 'rap', 'metal', 'folk', 'country', 'blues']
    for genre in genres:
        if genre in query_lower:
            entities.append(genre)
    
    # Extract time periods
    time_periods = ['7day', '1month', '3month', '6month', '12month', 'overall']
    for period in time_periods:
        if period in query_lower or period.replace('month', ' month') in query_lower:
            entities.append(period)
    
    return list(set(entities))  # Remove duplicates

# =============================================================================
# LANGGRAPH NODE FUNCTIONS
# =============================================================================

def classify_intent_node_hf(state: AgentState, hf_helper: HuggingFaceHelper, artist_names: List[str]) -> AgentState:
    """Classify user intent using Hugging Face"""
    
    try:
        # Use HF helper for classification
        intent, confidence = hf_helper.classify_intent(state["user_query"])
        
        # Extract entities
        entities = extract_entities(state["user_query"], artist_names)
        
        state["intent"] = intent
        state["confidence"] = confidence
        state["entities"] = entities
        
        print(f"🎯 Intent: {intent} (confidence: {confidence:.2f})")
        print(f"🏷️ Entities: {entities}")
        
    except Exception as e:
        state["error_message"] = f"Intent classification failed: {e}"
        state["intent"] = "LISTENING_ANALYSIS"  # Default fallback
        state["confidence"] = 0.5
        print(f"❌ Classification error: {e}")
    
    return state

def generate_artist_info_query(state: AgentState) -> AgentState:
    """Generate SQL query for artist information"""
    
    if not state["entities"]:
        state["error_message"] = "No artist name provided"
        return state
    
    artist_name = state["entities"][0]
    
    query = """
    SELECT 
        a.name as artist_name,
        a.bio_summary,
        a.listeners,
        a.playcount,
        GROUP_CONCAT(DISTINCT t.name) as tags,
        GROUP_CONCAT(DISTINCT sa.name) as similar_artists
    FROM artists a
    LEFT JOIN artist_tags at ON a.artist_id = at.artist_id
    LEFT JOIN tags t ON at.tag_id = t.tag_id
    LEFT JOIN artist_similar asim ON a.artist_id = asim.artist_id
    LEFT JOIN artists sa ON asim.similar_artist_id = sa.artist_id
    WHERE LOWER(a.name) LIKE LOWER(?)
    GROUP BY a.artist_id, a.name, a.bio_summary, a.listeners, a.playcount
    LIMIT 5
    """
    
    state["sql_query"] = query
    print(f"🔍 Generated artist info query for: {artist_name}")
    return state

def generate_recommendations_query(state: AgentState) -> AgentState:
    """Generate SQL query for recommendations"""
    
    if state["entities"]:
        # Specific artist recommendations
        artist_name = state["entities"][0]
        query = """
        SELECT DISTINCT
            sa.name as recommended_artist,
            sa.bio_summary,
            sa.listeners,
            asim.match_score,
            GROUP_CONCAT(DISTINCT t.name) as tags
        FROM artists a
        JOIN artist_similar asim ON a.artist_id = asim.artist_id
        JOIN artists sa ON asim.similar_artist_id = sa.artist_id
        LEFT JOIN artist_tags at ON sa.artist_id = at.artist_id
        LEFT JOIN tags t ON at.tag_id = t.tag_id
        WHERE LOWER(a.name) LIKE LOWER(?)
        GROUP BY sa.artist_id, sa.name, sa.bio_summary, sa.listeners, asim.match_score
        ORDER BY asim.match_score DESC
        LIMIT 10
        """
        print(f"🎵 Generating recommendations for: {artist_name}")
    else:
        # General recommendations based on user's top artists
        query = """
        SELECT DISTINCT
            sa.name as recommended_artist,
            sa.bio_summary,
            sa.listeners,
            asim.match_score,
            uta.rank as source_artist_rank,
            GROUP_CONCAT(DISTINCT t.name) as tags
        FROM user_top_artists uta
        JOIN artists a ON uta.artist_id = a.artist_id
        JOIN artist_similar asim ON a.artist_id = asim.artist_id
        JOIN artists sa ON asim.similar_artist_id = sa.artist_id
        LEFT JOIN artist_tags at ON sa.artist_id = at.artist_id
        LEFT JOIN tags t ON at.tag_id = t.tag_id
        WHERE uta.time_period = 'overall'
        GROUP BY sa.artist_id, sa.name, sa.bio_summary, sa.listeners, asim.match_score, uta.rank
        ORDER BY uta.rank ASC, asim.match_score DESC
        LIMIT 15
        """
        state["needs_api_call"] = True
        print("🎵 Generating general recommendations based on your listening history")
    
    state["sql_query"] = query
    return state

def generate_genre_exploration_query(state: AgentState) -> AgentState:
    """Generate SQL query for genre/tag exploration"""
    
    if not state["entities"]:
        # Show popular genres/tags
        query = """
        SELECT 
            t.name as tag_name,
            t.reach,
            t.taggings,
            t.wiki_summary,
            COUNT(DISTINCT at.artist_id) as artist_count
        FROM tags t
        JOIN artist_tags at ON t.tag_id = at.tag_id
        GROUP BY t.tag_id, t.name, t.reach, t.taggings, t.wiki_summary
        ORDER BY t.reach DESC
        LIMIT 20
        """
        print("🏷️ Exploring popular genres and tags")
    else:
        genre = state["entities"][0]
        query = """
        SELECT 
            a.name as artist_name,
            a.listeners,
            a.playcount,
            at.count as tag_relevance,
            a.bio_summary
        FROM tags t
        JOIN artist_tags at ON t.tag_id = at.tag_id
        JOIN artists a ON at.artist_id = a.artist_id
        WHERE LOWER(t.name) LIKE LOWER(?)
        ORDER BY at.count DESC, a.listeners DESC
        LIMIT 20
        """
        print(f"🏷️ Exploring genre: {genre}")
    
    state["sql_query"] = query
    return state

def generate_listening_analysis_query(state: AgentState) -> AgentState:
    """Generate SQL query for listening habit analysis"""
    
    # Extract number for LIMIT if provided
    limit = 10  # default
    for entity in state["entities"]:
        if entity.isdigit():
            limit = min(int(entity), 50)  # Cap at 50
            break
    
    query = f"""
    SELECT 
        a.name as artist_name,
        uta.playcount,
        uta.rank,
        GROUP_CONCAT(DISTINCT t.name) as tags
    FROM user_top_artists uta
    JOIN artists a ON uta.artist_id = a.artist_id
    LEFT JOIN artist_tags at ON a.artist_id = at.artist_id
    LEFT JOIN tags t ON at.tag_id = t.tag_id
    WHERE uta.time_period = 'overall'
    GROUP BY a.artist_id, a.name, uta.playcount, uta.rank
    ORDER BY uta.rank
    LIMIT {limit}
    """
    
    state["sql_query"] = query
    print(f"📊 Analyzing your top {limit} artists")
    return state

def execute_database_query(state: AgentState, db_manager: DatabaseManager) -> AgentState:
    """Execute the generated SQL query"""
    
    if not state["sql_query"]:
        state["error_message"] = "No SQL query generated"
        return state
    
    try:
        # Prepare parameters for query
        params = ()
        if state["entities"] and state["intent"] in ["ARTIST_INFO", "RECOMMENDATIONS", "GENRE_EXPLORATION"]:
            params = (f"%{state['entities'][0]}%",)
        
        results = db_manager.execute_query(state["sql_query"], params)
        state["query_results"] = results
        
        print(f"📊 Query executed successfully. Found {len(results)} results.")
        
        # Show first few results for debugging
        if results:
            print("Sample results:")
            for i, result in enumerate(results[:2]):
                print(f"  {i+1}. {list(result.keys())[:3]}...")
        
    except Exception as e:
        state["error_message"] = f"Database query execution failed: {e}"
        print(f"❌ Database query error: {e}")
    
    return state

def call_lastfm_api(state: AgentState, lastfm_api_key: str = None) -> AgentState:
    """Call Last.fm API for additional recommendations"""
    
    if not state.get("needs_api_call", False) or not lastfm_api_key:
        return state
    
    api_helper = LastFMAPIHelper(lastfm_api_key)
    
    try:
        api_results = []
        
        if state["intent"] == "RECOMMENDATIONS" and state["entities"]:
            artist_name = state["entities"][0]
            similar_artists = api_helper.get_similar_artists(artist_name)
            api_results.extend(similar_artists)
            print(f"📻 Found {len(similar_artists)} similar artists from Last.fm API")
        
        state["api_results"] = api_results
        
    except Exception as e:
        state["error_message"] = f"API call failed: {e}"
        print(f"❌ API call error: {e}")
    
    return state

def generate_response_hf(state: AgentState) -> AgentState:
    """Generate response using templates (no LLM needed)"""
    
    try:
        intent = state["intent"]
        results = state["query_results"]
        query = state["user_query"]
        
        if not results:
            state["response"] = "I couldn't find any results for your query. Try asking about your top artists or music recommendations!"
            return state
        
        # Generate response based on intent
        if intent == 'LISTENING_ANALYSIS':
            state["response"] = generate_listening_response(results, query)
        elif intent == 'RECOMMENDATIONS':
            state["response"] = generate_recommendations_response(results, query)
        elif intent == 'GENRE_EXPLORATION':
            state["response"] = generate_genre_response(results, query)
        elif intent == 'ARTIST_INFO':
            state["response"] = generate_artist_info_response(results, query)
        else:
            state["response"] = generate_default_response(results, query)
        
        print("✅ Response generated successfully")
        
    except Exception as e:
        state["error_message"] = f"Response generation failed: {e}"
        state["response"] = "I'm sorry, I encountered an error while processing your request."
        print(f"❌ Response generation error: {e}")
    
    return state

# =============================================================================
# RESPONSE GENERATORS
# =============================================================================

def generate_listening_response(results: List[Dict], query: str) -> str:
    """Generate listening analysis response"""
    
    if not results:
        return "No listening data found."
    
    # Check if this is the specific format from listening analysis query
    if 'item_name' in results[0] and 'analysis_type' in results[0]:
        response = f"🎵 **Your Top Artists:**\n\n"
        for i, row in enumerate(results[:10], 1):
            artist_name = row['item_name']
            playcount = row['playcount']
            tags = row.get('tags', '')
            tag_str = f" - Tags: {tags[:50]}..." if tags else ""
            response += f"{i}. **{artist_name}** ({playcount} plays){tag_str}\n"
        return response
    
    # Fallback for other listening analysis formats
    elif 'artist_name' in results[0]:
        response = f"🎵 **Your Top Artists:**\n\n"
        for i, row in enumerate(results[:10], 1):
            tags = row.get('tags', '')
            tag_str = f" ({tags[:50]}...)" if tags else ""
            response += f"{i}. **{row['artist_name']}** - {row['playcount']} plays{tag_str}\n"
    
    elif 'album_name' in results[0]:
        response = f"💿 **Your Top Albums:**\n\n"
        for i, row in enumerate(results[:10], 1):
            response += f"{i}. **{row['album_name']}** by {row['artist_name']} - {row['playcount']} plays\n"
    
    elif 'track_name' in results[0]:
        response = f"🎧 **Your Top Tracks:**\n\n"
        for i, row in enumerate(results[:10], 1):
            response += f"{i}. **{row['track_name']}** by {row['artist_name']} - {row['playcount']} plays\n"
    
    else:
        response = "Here's your listening data:\n\n"
        for row in results[:5]:
            response += f"• {list(row.values())[0]}\n"
    
    return response

def generate_recommendations_response(results: List[Dict], query: str) -> str:
    """Generate recommendations response"""
    
    if not results:
        return "No recommendations found."
    
    response = f"🎵 **Music Recommendations for You:**\n\n"
    
    for i, row in enumerate(results[:10], 1):
        if 'recommended_artist' in row:
            score = row.get('match_score', 0)
            tags = row.get('tags', '')
            based_on = row.get('based_on_artist', '')
            
            response += f"{i}. **{row['recommended_artist']}**"
            if based_on:
                response += f" (similar to {based_on})"
            if score:
                response += f" - {score:.1f}% match"
            if tags:
                response += f"\n   Tags: {tags[:100]}..."
            response += "\n\n"
    
    return response

def generate_genre_response(results: List[Dict], query: str) -> str:
    """Generate genre exploration response"""
    
    if not results:
        return "No genre data found."
    
    if 'artist_name' in results[0]:
        response = f"🏷️ **Artists in this Genre:**\n\n"
        for i, row in enumerate(results[:15], 1):
            response += f"{i}. **{row['artist_name']}** ({row['listeners']:,} listeners)\n"
    else:
        response = f"🏷️ **Popular Genres in Your Library:**\n\n"
        for i, row in enumerate(results[:15], 1):
            response += f"{i}. **{row['tag_name']}** - {row['artist_count']} artists\n"
    
    return response

def generate_artist_info_response(results: List[Dict], query: str) -> str:
    """Generate artist info response"""
    
    if not results:
        return "I couldn't find information about that artist."
    
    row = results[0]
    response = f"🎤 **{row['artist_name']}**\n\n"
    
    if row.get('bio_summary'):
        bio = row['bio_summary'][:300] + "..." if len(row['bio_summary']) > 300 else row['bio_summary']
        response += f"**Bio:** {bio}\n\n"
    
    if row.get('listeners'):
        response += f"**Listeners:** {row['listeners']:,}\n"
    
    if row.get('playcount'):
        response += f"**Total Plays:** {row['playcount']:,}\n"
    
    if row.get('tags'):
        response += f"**Tags:** {row['tags']}\n"
    
    if row.get('similar_artists'):
        response += f"**Similar Artists:** {row['similar_artists']}\n"
    
    return response

def generate_default_response(results: List[Dict], query: str) -> str:
    """Generate default response"""
    if not results:
        return "No results found for your query."
    return f"Here's what I found:\n\n" + "\n".join([str(row) for row in results[:5]])

# =============================================================================
# CHAT INTERFACE
# =============================================================================

class MusicChatInterface:
    """Interactive chat interface for the music discovery agent"""
    
    def __init__(self, agent):
        self.agent = agent
        self.conversation_history = []
    
    def process_query(self, user_input: str):
        """Process user query and return response"""
        print(f"\n🎵 Processing: '{user_input}'")
        print("=" * 60)
        
        # Initialize state
        initial_state = {
            "user_query": user_input,
            "intent": "",
            "confidence": 0.0,
            "entities": [],
            "sql_query": "",
            "query_results": [],
            "api_results": [],
            "response": "",
            "error_message": "",
            "needs_api_call": False
        }
        
        try:
            # Run the agent
            result = self.agent.invoke(initial_state)
            
            # Store conversation
            self.conversation_history.append((user_input, result.get("response", "No response generated")))
            
            # Display results
            print("\n" + "=" * 60)
            if result.get("error_message"):
                print(f"❌ Error: {result['error_message']}")
            else:
                print("🎵 Response:")
                print("-" * 40)
                print(result["response"])
            
            # Show debug info
            print(f"\n🔍 Debug Info:")
            print(f"Intent: {result.get('intent', 'Unknown')} (confidence: {result.get('confidence', 0):.2f})")
            print(f"Entities: {result.get('entities', [])}")
            if result.get("query_results"):
                print(f"Database results: {len(result['query_results'])} records")
            if result.get("api_results"):
                print(f"API results: {len(result['api_results'])} records")
            
            return result
            
        except Exception as e:
            error_msg = f"Agent execution failed: {e}"
            print(f"❌ {error_msg}")
            return {"error_message": error_msg}
    
    def show_examples(self):
        """Show example queries"""
        examples = {
            "Artist Information": [
                "Tell me about Radiohead",
                "What can you tell me about similar artists to The Beatles?",
                "Who is Bon Iver and what genre do they play?"
            ],
            "Recommendations": [
                "Recommend me artists similar to Bon Iver",
                "What new music should I listen to based on my taste?",
                "Find me artists like my top artists"
            ],
            "Genre Exploration": [
                "Show me indie rock artists",
                "What electronic artists do you have?",
                "Explore jazz music in my library"
            ],
            "Listening Analysis": [
                "Analyze my listening habits",
                "What are my top artists?",
                "Show me my listening patterns"
            ]
        }
        
        print("💡 Example Queries:")
        print("=" * 50)
        for category, queries in examples.items():
            print(f"\n{category}:")
            for i, query in enumerate(queries, 1):
                print(f"  {i}. {query}")