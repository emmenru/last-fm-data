# utils/music_discovery_helper.py
"""
Music Discovery AI Agent

Processes natural language queries about Last.fm music data using a hybrid approach:
- AI intent classification (SentenceTransformer)
- Hybrid entity extraction (regex + SentenceTransformer semantic AI)
- 7-node LangGraph workflow for clean architecture

Supports artist info, listening stats, and music recommendations.
Example: "Tell me about Radiohead" → artist biography and stats
"""
import re
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, TypedDict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.graph import StateGraph, START, END


# Constants
TIME_DISPLAY = {
    "7day": "past week", "1month": "past month", "3month": "past 3 months", 
    "6month": "past 6 months", "12month": "past year", "overall": "all time"
}

TIME_PERIODS = ["7day", "1month", "3month", "6month", "12month", "overall"]


class AgentState(TypedDict):
    """State for the music discovery agent."""
    user_input: str
    intent: str
    sql_query: str
    results: List[Dict]
    response: str
    query_metadata: Optional[Dict[str, Any]]


# Utility functions
def format_number(num: int, suffix: str) -> str:
    """Format large numbers with K/M suffixes."""
    if num >= 1000000:
        return f"{num/1000000:.1f}M {suffix}"
    elif num >= 1000:
        return f"{num/1000:.0f}K {suffix}"
    else:
        return f"{num:,} {suffix}"


def format_play_count(playcount: int) -> str:
    """Format play count for display."""
    return f"{playcount/1000:.0f}K" if playcount >= 1000 else str(playcount)


def clean_bio_text(bio: str) -> str:
    """Clean HTML and limit bio to 3 sentences."""
    bio_clean = re.sub(r'<[^>]+>', '', bio.strip())
    bio_clean = re.sub(r'^\d+\.\s*', '', bio_clean)
    sentences = bio_clean.split('.')
    return '. '.join(sentences[:3]) + '.' if len(sentences) > 3 else bio_clean


def get_match_rating(match_score: float) -> str:
    """Convert similarity score to rating."""
    if match_score >= 0.8: return "(Very High Match)"
    elif match_score >= 0.6: return "(High Match)"
    elif match_score >= 0.4: return "(Good Match)"
    else: return "(Similar)"


class DatabaseManager:
    """Handles database connections and queries."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def execute_query(self, sql_query: str) -> List[Dict]:
        """Execute SQL query and return results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(sql_query)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Query error: {e}")
            return []


class AIClassifier:
    """AI-powered intent classifier using sentence transformers."""
    
    def __init__(self):
        print("Loading AI model for intent classification...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Training examples for each intent
        self.training_examples = {
            "listening_stats": [
                "show me my top artists", "what are my most played songs",
                "my listening history", "top tracks this month", "my favorite albums",
                "what music do I listen to", "my top artists overall",
                "show me my tracks", "my music statistics", "top songs last week",
                "my artists", "my albums", "show me albums", "my tracks",
                "top artists 7day", "top songs 1month", "albums 3month"
            ],
            "artist_info": [
                "tell me about radiohead", "who is bob dylan", 
                "information about the beatles", "describe this artist",
                "what do you know about nirvana", "artist biography",
                "background on this band", "who are the rolling stones",
                "tell me about", "information about", "who is"
            ],
            "recommend_music": [
                "find artists similar to radiohead", "recommend music like jazz",
                "suggest artists", "music like the beatles", "discover new music",
                "artists similar to", "find me something like", "recommendations",
                "similar to", "artists like", "find artists", "suggest"
            ]
        }
        
        # Pre-compute embeddings for each intent
        print("Computing semantic embeddings...")
        self.intent_embeddings = {}
        for intent, examples in self.training_examples.items():
            embeddings = self.model.encode(examples)
            self.intent_embeddings[intent] = np.mean(embeddings, axis=0)
        
        print("AI classifier ready! 🤖")
    
    def classify(self, user_input: str) -> str:
        """Classify user input into intent category."""
        input_embedding = self.model.encode([user_input])
        
        similarities = {}
        for intent, intent_embedding in self.intent_embeddings.items():
            similarity = cosine_similarity(input_embedding, [intent_embedding])[0][0]
            similarities[intent] = similarity
        
        return max(similarities, key=similarities.get)


class FastEntityExtractor:
    """Fast entity extraction using sentence transformers + simple patterns."""
    
    def __init__(self, model: SentenceTransformer):
        print("Setting up fast entity extraction...")
        self.model = model
        
        # Semantic examples for time periods
        self.time_examples = {
            "7day": ["last week", "past week", "7 days", "recent week", "this week"],
            "1month": ["last month", "past month", "this month", "recent month"],
            "3month": ["3 months", "three months", "quarter", "past 3 months"],
            "6month": ["6 months", "six months", "half year", "past 6 months"], 
            "12month": ["year", "12 months", "past year", "last year", "annual"],
            "overall": ["all time", "overall", "total", "everything", "lifetime"]
        }
        
        # Semantic examples for content types
        self.content_examples = {
            "artists": ["artists", "bands", "musicians", "performers"],
            "tracks": ["songs", "tracks", "music", "tunes"],
            "albums": ["albums", "records", "releases"]
        }
        
        # Pre-compute embeddings for fast lookup
        self._setup_embeddings()
        print("Fast entity extractor ready! ⚡")
    
    def _setup_embeddings(self):
        """Pre-compute embeddings for semantic matching."""
        self.time_embeddings = {}
        for period, examples in self.time_examples.items():
            embeddings = self.model.encode(examples)
            self.time_embeddings[period] = np.mean(embeddings, axis=0)
        
        self.content_embeddings = {}
        for content_type, examples in self.content_examples.items():
            embeddings = self.model.encode(examples)
            self.content_embeddings[content_type] = np.mean(embeddings, axis=0)
    
    def extract_entities(self, user_input: str) -> Dict[str, Any]:
        """Extract entities using fast AI + regex patterns."""
        return {
            "artist_name": self._extract_artist_name(user_input),
            "time_period": self._extract_time_period_semantic(user_input),
            "content_type": self._extract_content_type_semantic(user_input)
        }
    
    def _extract_artist_name(self, text: str) -> Optional[str]:
        """Extract artist name using regex patterns."""
        patterns = [
            # Artist info patterns
            r'(?:tell me about|who is|about|information about)\s+(.+?)(?:[?.!]|$)',
            
            # Similarity patterns  
            r'(?:similar to|like|artists like|music like|find artists like)\s+(.+?)(?:[?.!]|$)',
            r'(?:find|recommend|suggest).*?(?:similar.*to|like)\s+(.+?)(?:[?.!]|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                
                # Cleanup - remove obvious non-artist words at the end
                candidate = re.sub(r'\s+(artist|band|musician|group)$', '', 
                                 candidate, flags=re.IGNORECASE).strip()
                
                if candidate and len(candidate) > 1:
                    return candidate
        
        return None
    
    def _extract_time_period_semantic(self, text: str) -> str:
        """Extract time period using semantic similarity."""
        # First check for exact matches (fastest)
        text_lower = text.lower()
        for period in TIME_PERIODS:
            if period in text_lower:
                return period
        
        # Then use semantic matching
        input_embedding = self.model.encode([text])
        
        best_score = 0
        best_period = "overall"
        
        for period, period_embedding in self.time_embeddings.items():
            similarity = cosine_similarity(input_embedding, [period_embedding])[0][0]
            
            if similarity > best_score:
                best_score = similarity
                best_period = period
        
        # Only return non-default if confidence is reasonable
        return best_period if best_score > 0.3 else "overall"
    
    def _extract_content_type_semantic(self, text: str) -> str:
        """Extract content type using semantic similarity."""
        # First check for exact matches (fastest)
        text_lower = text.lower()
        if any(word in text_lower for word in ["song", "songs", "track", "tracks"]):
            return "tracks"
        elif any(word in text_lower for word in ["album", "albums"]):
            return "albums"
        
        # Then use semantic matching
        input_embedding = self.model.encode([text])
        
        best_score = 0
        best_type = "artists"
        
        for content_type, content_embedding in self.content_embeddings.items():
            similarity = cosine_similarity(input_embedding, [content_embedding])[0][0]
            
            if similarity > best_score:
                best_score = similarity
                best_type = content_type
        
        # Only return non-default if confidence is reasonable
        return best_type if best_score > 0.3 else "artists"


class SQLGenerator:
    """Generates SQL queries using pre-extracted entities."""
    
    def generate_artist_sql(self, entities: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL for artist information using extracted entities."""
        artist_name = entities["artist_name"]
        
        if artist_name:
            sql = f"""
                SELECT name, bio_summary, listeners, playcount 
                FROM artists 
                WHERE LOWER(name) LIKE LOWER('%{artist_name}%') 
                LIMIT 5
            """
            metadata = {"query_type": f"artist_info_{artist_name}"}
        else:
            sql = ""
            metadata = {"error": "No artist found"}
        return sql, metadata
    
    def generate_recommendation_sql(self, entities: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL for music recommendations using extracted entities."""
        artist_name = entities["artist_name"]
        
        if artist_name:
            sql = f"""
                SELECT a2.name as similar_artist, ars.match_score 
                FROM artists a1 
                JOIN artist_similar ars ON a1.artist_id = ars.artist_id 
                JOIN artists a2 ON ars.similar_artist_id = a2.artist_id 
                WHERE LOWER(a1.name) LIKE LOWER('%{artist_name}%') 
                ORDER BY ars.match_score DESC LIMIT 10
            """
            metadata = {"query_type": f"recommend_similar_{artist_name}"}
        else:
            sql = ""
            metadata = {"error": "No target artist found"}
        return sql, metadata
    
    def generate_stats_sql(self, entities: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL for listening statistics using extracted entities."""
        content_type = entities["content_type"]
        time_period = entities["time_period"]
        
        # Map content types to table info
        table_map = {
            "tracks": ("user_top_tracks", "tracks", "track_id", "t"),
            "albums": ("user_top_albums", "albums", "album_id", "a"),
            "artists": ("user_top_artists", "artists", "artist_id", "a")
        }
        
        user_table, item_table, id_field, alias = table_map[content_type]
        
        if content_type == "artists":
            sql = f"""
                SELECT {alias}.name, u.playcount 
                FROM {user_table} u 
                JOIN {item_table} {alias} ON u.artist_id = {alias}.artist_id 
                WHERE u.time_period = '{time_period}'
                ORDER BY u.playcount DESC LIMIT 10
            """
        else:
            sql = f"""
                SELECT {alias}.name, ar.name as artist_name, u.playcount 
                FROM {user_table} u 
                JOIN {item_table} {alias} ON u.{id_field} = {alias}.{id_field} 
                JOIN artists ar ON {alias}.artist_id = ar.artist_id
                WHERE u.time_period = '{time_period}'
                ORDER BY u.playcount DESC LIMIT 10
            """
        
        metadata = {"query_type": f"listening_stats_{content_type}_{time_period}"}
        return sql, metadata


class ResponseGenerator:
    """Generates formatted responses."""
    
    def generate_response(self, state: AgentState) -> str:
        """Generate response based on intent and results."""
        if not state["results"]:
            return self._handle_no_results(state["intent"], state.get("query_metadata", {}))
        
        intent = state["intent"]
        results = state["results"]
        
        if intent == "artist_info":
            return self._format_artist_info(results[0])
        elif intent == "listening_stats":
            return self._format_listening_stats(results, state.get("query_metadata", {}))
        elif intent == "recommend_music":
            return self._format_recommendations(results, state.get("query_metadata", {}))
        
        return f"Found {len(results)} results."
    
    def _format_artist_info(self, artist: Dict[str, Any]) -> str:
        """Format artist information response."""
        name = artist.get('name', 'Unknown Artist')
        bio = artist.get('bio_summary', '')
        listeners = artist.get('listeners', 0)
        playcount = artist.get('playcount', 0)
        
        response = f"**{name}**\n\n"
        
        if bio and len(bio.strip()) > 10:
            bio_display = clean_bio_text(bio)
            response += f"{bio_display}\n\n"
        
        # Format stats
        stats = []
        if listeners > 0:
            stats.append(format_number(listeners, "listeners"))
        if playcount > 0:
            stats.append(format_number(playcount, "plays"))
        
        if stats:
            response += f"Stats: {' • '.join(stats)}"
        
        return response
    
    def _format_listening_stats(self, results: List[Dict], metadata: Dict) -> str:
        """Format listening statistics response."""
        query_type = metadata.get('query_type', '')
        
        # Extract content type and time period
        content_type = "artists"
        if 'tracks' in query_type:
            content_type = "tracks"
        elif 'albums' in query_type:
            content_type = "albums"
        
        time_period = next((p for p in TIME_PERIODS if p in query_type), "overall")
        time_display = TIME_DISPLAY.get(time_period, time_period)
        
        response = f"**Your top {content_type} ({time_display}):**\n\n"
        
        for i, item in enumerate(results[:10], 1):
            name = item.get('name', 'Unknown')
            playcount = item.get('playcount', 0)
            
            # Add artist name for tracks/albums
            if content_type in ['tracks', 'albums'] and 'artist_name' in item:
                artist = item.get('artist_name', '')
                if artist:
                    name = f"{name} by {artist}"
            
            play_str = format_play_count(playcount)
            response += f"{i:2d}. {name} ({play_str} plays)\n"
        
        total_plays = sum(item.get('playcount', 0) for item in results)
        response += f"\nTotal: {total_plays:,} plays"
        
        return response
    
    def _format_recommendations(self, results: List[Dict], metadata: Dict) -> str:
        """Format recommendations response."""
        query_type = metadata.get('query_type', '')
        target_artist = "that artist"
        
        if 'recommend_similar_' in query_type:
            target_artist = query_type.replace('recommend_similar_', '').replace('_', ' ').title()
        
        response = f"**Artists similar to {target_artist}:**\n\n"
        
        if not results:
            return f"No similar artists found for {target_artist} in your database."
        
        for i, item in enumerate(results[:8], 1):
            artist_name = item.get('similar_artist', item.get('name', 'Unknown'))
            match_score = item.get('match_score', 0)
            rating = get_match_rating(match_score)
            response += f"{i}. {artist_name} {rating}\n"
        
        response += f"\nFound {len(results)} similar artists in your database"
        return response
    
    def _handle_no_results(self, intent: str, metadata: Dict) -> str:
        """Handle cases where no results were found."""
        error_messages = {
            "artist_info": "That artist wasn't found in your music database.",
            "listening_stats": "No listening statistics found for that time period.",
            "recommend_music": "No similar artists found in your database."
        }
        
        error = metadata.get('error', '')
        if "No artist found" in error or "No target artist found" in error:
            example = "Radiohead" if intent != "listening_stats" else "my top artists"
            return f"I couldn't identify the artist. Try: '{example}'"
        
        return error_messages.get(intent, "No results found.")


class MusicDiscoveryAgent:
    """Main agent."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.classifier = AIClassifier()
        self.entity_extractor = FastEntityExtractor(self.classifier.model)  # Reuse same SentenceTransformer model
        self.response_generator = ResponseGenerator()
        self.sql_generator = SQLGenerator()
        self.session_history = []
        self.graph = self._create_graph()
    
    def _create_graph(self):
        """Create the workflow graph with proper node separation."""
        graph = StateGraph(AgentState)
        
        # Add all the workflow nodes
        nodes = [
            ("classify_intent", self._classify_intent),
            ("extract_entities", self._extract_entities),
            ("get_bio_info", self._generate_artist_sql),
            ("get_recommendation", self._generate_recommendation_sql),
            ("describe_my_listening", self._generate_stats_sql),
            ("execute_sql_query", self._execute_query),
            ("generate_response", self._generate_response)
        ]
        
        for name, func in nodes:
            graph.add_node(name, func)
        
        # Connect the nodes
        graph.add_edge(START, "classify_intent")
        graph.add_edge("classify_intent", "extract_entities")
        graph.add_conditional_edges(
            "extract_entities", self._route_sql_type,
            {"get_bio_info": "get_bio_info", 
             "get_recommendation": "get_recommendation", 
             "describe_my_listening": "describe_my_listening"}
        )
        
        for node in ["get_bio_info", "get_recommendation", "describe_my_listening"]:
            graph.add_edge(node, "execute_sql_query")
        
        graph.add_edge("execute_sql_query", "generate_response")
        graph.add_edge("generate_response", END)
        
        return graph.compile()
    
    def _classify_intent(self, state: AgentState) -> AgentState:
        """Classify the user's intent using SentenceTransformer semantic similarity."""
        state["intent"] = self.classifier.classify(state["user_input"])
        return state
    
    def _extract_entities(self, state: AgentState) -> AgentState:
        """Extract entities using hybrid approach: regex for artists, SentenceTransformer for time/content."""
        entities = self.entity_extractor.extract_entities(state["user_input"])
        # Store entities in query_metadata for later use
        state["query_metadata"] = {"entities": entities}
        return state
    
    def _route_sql_type(self, state: AgentState) -> str:
        """Route to the right SQL generator based on intent."""
        routing = {"artist_info": "get_bio_info", "listening_stats": "describe_my_listening"}
        return routing.get(state["intent"], "get_recommendation")
    
    def _generate_artist_sql(self, state: AgentState) -> AgentState:
        """Generate SQL for artist queries."""
        entities = state["query_metadata"]["entities"]
        sql, metadata = self.sql_generator.generate_artist_sql(entities)
        state["sql_query"] = sql
        # Merge metadata
        state["query_metadata"].update(metadata)
        return state
    
    def _generate_recommendation_sql(self, state: AgentState) -> AgentState:
        """Generate SQL for recommendation queries."""
        entities = state["query_metadata"]["entities"]
        sql, metadata = self.sql_generator.generate_recommendation_sql(entities)
        state["sql_query"] = sql
        # Merge metadata
        state["query_metadata"].update(metadata)
        return state
    
    def _generate_stats_sql(self, state: AgentState) -> AgentState:
        """Generate SQL for stats queries."""
        entities = state["query_metadata"]["entities"]
        sql, metadata = self.sql_generator.generate_stats_sql(entities)
        state["sql_query"] = sql
        # Merge metadata
        state["query_metadata"].update(metadata)
        return state
    
    def _execute_query(self, state: AgentState) -> AgentState:
        """Execute the SQL query."""
        try:
            if state["sql_query"]:
                results = self.db_manager.execute_query(state["sql_query"])
                state["results"] = results[:10]
            else:
                state["results"] = []
        except Exception as e:
            print(f"Database error: {e}")
            state["results"] = []
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate the final response."""
        state["response"] = self.response_generator.generate_response(state)
        return state
    
    def process_query(self, user_input: str) -> Dict[str, Any]:
        """Process a user query and return the result."""
        state = {
            "user_input": user_input,
            "intent": "",
            "sql_query": "",
            "results": [],
            "response": "",
            "query_metadata": {}
        }
        
        try:
            final_state = self.graph.invoke(state)
            self.session_history.append({
                "user_input": user_input,
                "intent": final_state["intent"],
                "response": final_state["response"],
                "timestamp": datetime.now().isoformat()
            })
            return final_state
        except Exception as e:
            print(f"Error: {e}")
            return {**state, "response": "Sorry, I encountered an error."}


def create_music_agent(db_path: str) -> MusicDiscoveryAgent:
    """Create a music discovery agent."""
    return MusicDiscoveryAgent(DatabaseManager(db_path))