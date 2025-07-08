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
    """Custom state class for our music discovery agent. Container that carriers info between different stpes of the workflow (workflow orchestration). Uses a state-passing pattern based on nodes."""
    user_query: str # User question 
    intent: str # Classified intent 
    confidence: float
    entities: List[str] # Extracted entities
    sql_query: str
    query_params: tuple 
    query_results: List[Dict[str, Any]] # Output from database query 
    api_results: List[Dict[str, Any]] # Output from last.fm API query 
    response: str # Formatted output message 
    error_message: str
    needs_api_call: bool # Whether or not to call last.fm API 

# =============================================================================
# ENTITY EXTRACTION (EXTRACT INFO FROM USER TEXT)
# =============================================================================

def extract_entities(query: str, artist_names: List[str]) -> List[str]:
    """Extract entities from user query"""
    entities = []
    query_lower = query.lower()
    
    # Extract numbers needed for SQL query 
    numbers = re.findall(r'\b(\d+)\b', query_lower) # "What are my top 5 artists?" -> ['5']
    entities.extend(numbers)
    
    # Extract artist names (check against database)
    for artist_name in artist_names:
        if artist_name in query_lower and len(artist_name) > 3:
            entities.append(artist_name)
            break  # Take first match
    
    # Extract common genres to be able to filter by musical genre in our queries
    genres = ['indie', 'rock', 'pop', 'jazz', 'electronic', 'classical', 
             'hip hop', 'rap', 'metal', 'folk', 'country', 'blues']
    for genre in genres:
        if genre in query_lower:
            entities.append(genre)
    
    # Extract time periods to be able to filter on different time periods for top artists/albums/tracks in the database
    time_periods = ['7day', '1month', '3month', '6month', '12month', 'overall']
    for period in time_periods:
        if period in query_lower or period.replace('month', ' month') in query_lower:
            entities.append(period)
    
    return list(set(entities))  # Remove duplicates


# =============================================================================
# DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    """Custom helper class to handle database operaions. Wrapper around SQLite."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute generic SQL query and return results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()] # Return list of dictionaries
        except Exception as e:
            print(f"Database error: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test database connection and return success status"""
        try:
            tables = self.execute_query("SELECT name FROM sqlite_master WHERE type='table'")
            print(f"Database: {len(tables)} tables")
            return len(tables) > 0
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False
    
    def get_artist_names(self, limit: int = 200) -> List[str]:
        """Get list of artist names for entity extraction"""
        try:
            query = "SELECT DISTINCT name FROM artists ORDER BY listeners DESC LIMIT ?"
            results = self.execute_query(query, (limit,))
            artist_names = [row['name'].lower() for row in results]
            print(f"Loaded {len(artist_names)} artist names")
            return artist_names
        except Exception as e:
            print(f"Could not load artist names: {e}")
            return []

# =============================================================================
# HUGGING FACE HELPER
# =============================================================================

class HuggingFaceHelper:
    """Helper class for Hugging Face API calls. Used specifically to classify user intents using AI."""
    
    def __init__(self, hf_token: str = None):
        # Setup API credentials and endpoints 
        self.hf_token = hf_token #Free token 
        self.headers = {}
        if hf_token:
            self.headers["Authorization"] = f"Bearer {hf_token}"
        
        # API endpoints
        # Use Facebook's model for zero-shot classification
        self.classification_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
        
        # Intent labels for classification
        self.intent_labels = [
            "listening analysis and user statistics", # What are my top artists? 
            "music recommendations and suggestions",  # Recommend similar artists
            "genre and tag exploration", # Show me indie rock artists 
            "artist information and biography" # Tell me about Radiohead
        ]
    
    def classify_intent(self, query: str) -> Tuple[str, float]:
        """Classify intent using HF zero-shot classification"""
        payload = {
            "inputs": query,
            "parameters": {
                "candidate_labels": self.intent_labels
            }
        }
        
        try: # Send request to Hugging Face 
            response = requests.post(
                self.classification_url, 
                headers=self.headers, 
                json=payload,
                timeout=15
            )
            
            # Process the response 
            if response.status_code == 200:
                result = response.json()
                
                # Handle potential error responses
                if isinstance(result, dict) and 'error' in result:
                    print(f"HF API error: {result['error']}")
                    return self._fallback_classification(query)
                
                best_label = result['labels'][0] # Most likely category
                confidence = result['scores'][0] # How confident in range 0-1
                
                # Map to our system's intent names
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
    """Handles Last.fm API calls for extended recommendations. Used to fetch data hat is not in the users database."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://ws.audioscrobbler.com/2.0/"
    
    def get_similar_artists(self, artist_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get similar artists from Last.fm API (globally)"""
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
            # Parse the nested response structure (JSON)
            if 'similarartists' in data and 'artist' in data['similarartists']:
                return data['similarartists']['artist']
            return [] # API succeeded but no results 
        except Exception as e:
            print(f"Last.fm API error: {e}")
            return [] # API failed, return empty list 
    
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
# LANGGRAPH NODE FUNCTIONS - TO DO: Go through and comment
# =============================================================================

def classify_intent_node_hf(state: AgentState, hf_helper: HuggingFaceHelper, artist_names: List[str]) -> AgentState:
    """Classify user intent using Hugging Face"""
    try:
        intent, confidence = hf_helper.classify_intent(state["user_query"])
        entities = extract_entities(state["user_query"], artist_names)
        
        state.update({
            "intent": intent,
            "confidence": confidence,
            "entities": entities
        })
        print(f"Intent: {intent} (confidence: {confidence:.2f}) | Entities: {entities}")
        
    except Exception as e:
        state.update({
            "intent": "LISTENING_ANALYSIS",
            "confidence": 0.5,
            "error_message": f"Intent classification failed: {e}"
        })
        print(f"Classification error: {e}")
    
    return state

def generate_sql_query(state: AgentState) -> AgentState:
    """Generate SQL query based on intent and entities"""
    intent = state["intent"]
    entities = state["entities"]
    
    # Extract common parameters
    limit = next((int(e) for e in entities if e.isdigit()), 10)
    limit = min(limit, 50)  # Cap at 50
    
    # FIXED: Use any non-numeric entity instead of checking against artist_names
    text_entities = [e for e in entities if not e.isdigit()]
    first_entity = text_entities[0] if text_entities else None

    print(f"🔍 DEBUG - text_entities: {text_entities}")
    print(f"🔍 DEBUG - first_entity: {first_entity}")
    
    # SQL query templates
    queries = {
        "ARTIST_INFO": """
            SELECT a.name as artist_name, a.bio_summary, a.listeners, a.playcount,
                   GROUP_CONCAT(DISTINCT t.name) as tags, GROUP_CONCAT(DISTINCT sa.name) as similar_artists
            FROM artists a
            LEFT JOIN artist_tags at ON a.artist_id = at.artist_id
            LEFT JOIN tags t ON at.tag_id = t.tag_id
            LEFT JOIN artist_similar asim ON a.artist_id = asim.artist_id
            LEFT JOIN artists sa ON asim.similar_artist_id = sa.artist_id
            WHERE LOWER(a.name) LIKE LOWER(?)
            GROUP BY a.artist_id LIMIT 5
        """,
        
        "RECOMMENDATIONS": """
            SELECT DISTINCT sa.name as recommended_artist, sa.listeners, asim.match_score
            FROM artists a
            JOIN artist_similar asim ON a.artist_id = asim.artist_id
            JOIN artists sa ON asim.similar_artist_id = sa.artist_id
            WHERE LOWER(a.name) LIKE LOWER(?)
            ORDER BY asim.match_score DESC LIMIT 10
        """ if first_entity else """
            SELECT DISTINCT sa.name as recommended_artist, sa.listeners, asim.match_score, a.name as based_on_artist
            FROM user_top_artists uta
            JOIN artists a ON uta.artist_id = a.artist_id
            JOIN artist_similar asim ON a.artist_id = asim.artist_id
            JOIN artists sa ON asim.similar_artist_id = sa.artist_id
            WHERE uta.time_period = 'overall' AND uta.rank <= 10
            ORDER BY uta.rank ASC, asim.match_score DESC LIMIT 15
        """,
        
        "GENRE_EXPLORATION": """
            SELECT a.name as artist_name, a.listeners, at.count as tag_relevance
            FROM tags t
            JOIN artist_tags at ON t.tag_id = at.tag_id
            JOIN artists a ON at.artist_id = a.artist_id
            WHERE LOWER(t.name) LIKE LOWER(?)
            ORDER BY at.count DESC, a.listeners DESC LIMIT 20
        """ if first_entity else """
            SELECT t.name as tag_name, t.reach, COUNT(DISTINCT at.artist_id) as artist_count
            FROM tags t
            JOIN artist_tags at ON t.tag_id = at.tag_id
            GROUP BY t.tag_id, t.name, t.reach
            ORDER BY t.reach DESC LIMIT 20
        """,
        
        "LISTENING_ANALYSIS": f"""
            SELECT a.name as artist_name, uta.playcount, uta.rank, GROUP_CONCAT(DISTINCT t.name) as tags
            FROM user_top_artists uta
            JOIN artists a ON uta.artist_id = a.artist_id
            LEFT JOIN artist_tags at ON a.artist_id = at.artist_id
            LEFT JOIN tags t ON at.tag_id = t.tag_id
            WHERE uta.time_period = 'overall'
            GROUP BY a.artist_id ORDER BY uta.rank LIMIT {limit}
        """
    }
    
    state["sql_query"] = queries.get(intent, queries["LISTENING_ANALYSIS"])
    
    # FIXED: Set query parameters properly
    if intent in ["ARTIST_INFO", "RECOMMENDATIONS", "GENRE_EXPLORATION"] and first_entity:
        state["query_params"] = (f"%{first_entity}%",)
    else:
        state["query_params"] = ()
    
    # Set API call flag
    if intent == "RECOMMENDATIONS" and not first_entity:
        state["needs_api_call"] = True
    
    
    #print(f"DEBUG - query_params: {state.get('query_params', 'NOT SET')}")

    print(f"Generated {intent.lower().replace('_', ' ')} query")
    
    return state

def execute_database_query(state: AgentState, db_manager: DatabaseManager) -> AgentState:
    """Execute the generated SQL query"""
    if not state["sql_query"]:
        state["error_message"] = "No SQL query generated"
        return state
    
    try:
        params = state.get("query_params", ())
        results = db_manager.execute_query(state["sql_query"], params)
        state["query_results"] = results
        print(f"Found {len(results)} results")
        
    except Exception as e:
        state["error_message"] = f"Database query failed: {e}"
        print(f"Database error: {e}")
    
    return state

def call_lastfm_api(state: AgentState, lastfm_api_key: str = None) -> AgentState:
    """Call Last.fm API for additional recommendations"""
    if not state.get("needs_api_call", False) or not lastfm_api_key:
        return state
    
    try:
        api_helper = LastFMAPIHelper(lastfm_api_key)
        api_results = []
        
        if state["intent"] == "RECOMMENDATIONS" and state["entities"]:
            artist_name = state["entities"][0]
            similar_artists = api_helper.get_similar_artists(artist_name)
            api_results.extend(similar_artists)
            print(f"📻 Found {len(similar_artists)} similar artists from Last.fm API")
        
        state["api_results"] = api_results
        
    except Exception as e:
        state["error_message"] = f"API call failed: {e}"
        print(f"API error: {e}")
    
    return state

def generate_response_hf(state: AgentState) -> AgentState:
    """Generate response using templates"""
    try:
        if not state["query_results"]:
            state["response"] = "I couldn't find any results for your query. Try asking about your top artists or music recommendations!"
            return state
        
        state["response"] = generate_response_by_intent(
            state["intent"], 
            state["query_results"], 
            state["user_query"]
        )
        print("Response generated")
        
    except Exception as e:
        state["error_message"] = f"Response generation failed: {e}"
        state["response"] = "I'm sorry, I encountered an error while processing your request."
        print(f"Response error: {e}")
    
    return state

# =============================================================================
# RESPONSE GENERATORS - TO DO: Go through and comment
# =============================================================================

def generate_response_by_intent(intent: str, results: List[Dict], query: str) -> str:
    """Unified response generator using templates"""
    
    if not results:
        return "No results found for your query. Try asking about your top artists or music recommendations!"
    
    # Response templates
    templates = {
        'LISTENING_ANALYSIS': {
            'title': '🎵 **Your Top Artists:**',
            'format': lambda i, row: f"{i}. **{row.get('artist_name', row.get('item_name', 'Unknown'))}** ({row.get('playcount', 0)} plays){_format_tags(row.get('tags', ''))}"
        },
        'RECOMMENDATIONS': {
            'title': '🎵 **Music Recommendations for You:**',
            'format': lambda i, row: f"{i}. **{row.get('recommended_artist', 'Unknown')}**{_format_score(row.get('match_score', 0))}{_format_tags(row.get('tags', ''))}"
        },
        'GENRE_EXPLORATION': {
            'title': '🏷️ **Artists in this Genre:**' if 'artist_name' in results[0] else '🏷️ **Popular Genres:**',
            'format': lambda i, row: f"{i}. **{row.get('artist_name', row.get('tag_name', 'Unknown'))}** ({row.get('listeners', row.get('artist_count', 0)):,} {'listeners' if 'listeners' in row else 'artists'})"
        },
        'ARTIST_INFO': {
            'title': f"🎤 **{results[0].get('artist_name', 'Artist')}**",
            'format': lambda i, row: _format_artist_info(row)
        }
    }
    
    template = templates.get(intent)
    if not template:
        return f"Here's what I found:\n\n" + "\n".join([f"• {list(row.values())[0]}" for row in results[:5]])
    
    # Generate response
    response = f"{template['title']}\n\n"
    
    if intent == 'ARTIST_INFO':
        return template['format'](0, results[0])
    
    for i, row in enumerate(results[:15], 1):
        response += template['format'](i, row) + "\n"
    
    return response

def _format_tags(tags: str) -> str:
    """Format tags for display"""
    return f" - Tags: {tags[:50]}..." if tags and len(tags) > 3 else ""

def _format_score(score) -> str:
    """Format match score for display"""
    return f" - {float(score):.0f}% match" if score else ""

def _format_artist_info(row: Dict) -> str:
    """Format artist information"""
    response = f"🎤 **{row.get('artist_name', 'Unknown Artist')}**\n\n"
    
    if row.get('bio_summary'):
        bio = row['bio_summary'][:300] + "..." if len(row['bio_summary']) > 300 else row['bio_summary']
        response += f"**Bio:** {bio}\n\n"
    
    info_items = [
        ('listeners', 'Listeners'),
        ('playcount', 'Total Plays'),
        ('tags', 'Tags'),
        ('similar_artists', 'Similar Artists')
    ]
    
    for key, label in info_items:
        if row.get(key):
            value = f"{row[key]:,}" if key in ['listeners', 'playcount'] else row[key]
            response += f"**{label}:** {value}\n"
    
    return response

# Legacy functions for backward compatibility (delegates to unified function)
def generate_listening_response(results: List[Dict], query: str) -> str:
    return generate_response_by_intent('LISTENING_ANALYSIS', results, query)

def generate_recommendations_response(results: List[Dict], query: str) -> str:
    return generate_response_by_intent('RECOMMENDATIONS', results, query)

def generate_genre_response(results: List[Dict], query: str) -> str:
    return generate_response_by_intent('GENRE_EXPLORATION', results, query)

def generate_artist_info_response(results: List[Dict], query: str) -> str:
    return generate_response_by_intent('ARTIST_INFO', results, query)

def generate_default_response(results: List[Dict], query: str) -> str:
    return generate_response_by_intent('DEFAULT', results, query)

# =============================================================================
# CHAT INTERFACE - TO DO: Go through and comment
# =============================================================================

class MusicChatInterface:
    """Simple chat interface for the music discovery agent"""
    
    def __init__(self, agent):
        self.agent = agent
        self.history = []
    
    def ask(self, query: str) -> Dict[str, Any]:
        """Process query and return result"""
        print(f"\n🎵 {query}")
        print("=" * 60)
        
        initial_state = {"user_query": query, "intent": "", "confidence": 0.0, "entities": [], 
        "sql_query": "", "query_params": (), "query_results": [], "api_results": [], 
        "response": "", "error_message": "", "needs_api_call": False
        }
 
        try:
            result = self.agent.invoke(initial_state)
            self.history.append((query, result.get("response", "")))
            
            if result.get("error_message"):
                print(f"{result['error_message']}")
            else:
                print(f"🎵 {result['response']}")
            
            return result
        except Exception as e:
            print(f"Error: {e}")
            return {"error_message": str(e)}
    
    def examples(self):
        """Show example queries"""
        examples = [
            "What are my top 5 artists?",
            "Recommend artists similar to Radiohead", 
            "Show me indie rock artists",
            "Tell me about The Beatles"
        ]
        print("💡 Try these examples:")
        for i, ex in enumerate(examples, 1):
            print(f"  {i}. {ex}")

# Simple standalone function alternative
def ask_music_agent(agent, query: str):
    """Simple function to ask the music agent"""
    return MusicChatInterface(agent).ask(query)