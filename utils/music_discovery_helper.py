# utils/music_discovery_helper.py
"""
Simplified Music Discovery AI Agent - Pattern-Based Classification
"""

import sqlite3
import re
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from datetime import datetime
from langgraph.graph import StateGraph, START, END

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# =============================================================================
# CONSTANTS
# =============================================================================

ARTIST_PATTERNS = [
    r'(?:tell me about|about|who is|information about)\s+(?:the\s+)?(?:artist|band|musician)\s+([^?.!]+?)(?:[?.!]|$)',
    r'(?:tell me about|about|who is|information about)\s+([^?.!]+?)(?:\s+(?:artist|band|musician))?(?:[?.!]|$)',
    r'(?:the\s+)?(?:artist|band|musician)\s+([^?.!]+?)(?:[?.!]|$)',
    r'biography(?:\s+for|\s+of)?\s+([^?.!]+?)(?:[?.!]|$)'
]

SIMILARITY_PATTERNS = [
    r'(?:similar to|like|artists like|music like|bands like)\s+([^?.!]+?)(?:[?.!]|$)',
    r'(?:find|recommend|suggest).*(?:similar.*to|like)\s+([^?.!]+?)(?:[?.!]|$)',
    r'(?:based on|inspired by)\s+([^?.!]+?)(?:[?.!]|$)'
]

CONTENT_TYPE_KEYWORDS = {
    "tracks": ["song", "songs", "track", "tracks"],
    "albums": ["album", "albums"]
}

TIME_PERIODS = ["7day", "1month", "3month", "6month", "overall"]

CLEANUP_WORDS = {
    "artist": r'\b(the|a|an|is|are|was|were|artist|artists|band|bands|musician|musicians)\b',
    "similarity": r'\b(the|a|an|artist|artists|band|bands|musician|musicians)\b'
}

# =============================================================================
# CORE CLASSES
# =============================================================================

class AgentState(TypedDict):
    user_input: str
    intent: str
    sql_query: str
    results: List[Dict]
    response: str
    query_metadata: Optional[Dict[str, Any]]

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def execute_query(self, sql_query: str) -> List[Dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(sql_query)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Query error: {e}")
            return []


class PatternBasedClassifier:
    """Simple, fast pattern-based intent classification"""
    
    def __init__(self):
        self.classification_patterns = {
            "listening_stats": [
                "my top", "my most", "show me my", "my listening", "my statistics", 
                "my personal", "my history", "give my", "top artists", "top tracks", 
                "top songs", "top albums", "my favorites", "what I listen to",
                "my music", "show me popular"
            ],
            "artist_info": [
                "tell me about", "who is", "information about", "biography for",
                "what do you know about", "describe", "background on"
            ],
            "recommend_music": [
                "recommend", "suggest", "discover", "find artists", "similar to", 
                "music like", "artists like", "mood", "energetic", "new sounds",
                "something", "find me"
            ]
        }
    
    def classify(self, user_input: str) -> str:
        """Classify user intent using pattern matching"""
        query_lower = user_input.lower()
        
        # Check patterns in order of specificity
        for intent, phrases in self.classification_patterns.items():
            if any(phrase in query_lower for phrase in phrases):
                print(f"  Pattern match: {intent}")
                return intent
        
        # Default fallback for unmatched queries
        print(f"  No pattern match - defaulting to recommend_music")
        return "recommend_music"


class SimpleTextGenerator:
    def __init__(self, generator):
        self.generator = generator
    
    def invoke(self, prompt_text: str) -> str:
        try:
            responses = self.generator(
                prompt_text,
                max_new_tokens=80,           
                min_length=12,               
                do_sample=True,
                temperature=0.8,             
                top_p=0.85,                  
                top_k=35,                    
                repetition_penalty=1.2,     
                length_penalty=1.0,
                return_full_text=False,
                clean_up_tokenization_spaces=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            if not responses:
                return None
            
            response = responses[0]['generated_text'].strip()
            
            # Quality check
            if len(response) < 8 or self._is_poor_quality(response): 
                return None
                
            return response
            
        except Exception as e:
            print(f"Generation error: {e}")
            return None
    
    def _is_poor_quality(self, response: str) -> bool:
        poor_indicators = [
            "here are your music results",
            "give a friendly",
            response.count(" ") < 2,
            len(set(response.split())) < 2
        ]
        return any(indicator for indicator in poor_indicators)


class SQLGenerator:
    def __init__(self):
        self.cache = {}
        self.query_history = []
    
    def _get_cached_or_generate(self, cache_key: str, generator_func) -> Tuple[str, Dict[str, Any]]:
        if cache_key in self.cache:
            sql, metadata = self.cache[cache_key]
            print(f"  Using cached query: {metadata.get('query_type', 'unknown')}")
            return sql, metadata
        
        sql, metadata = generator_func()
        metadata["timestamp"] = datetime.now().isoformat()
        self.cache[cache_key] = (sql, metadata)
        self.query_history.append({"sql": sql, "metadata": metadata})
        
        return sql, metadata
    
    def generate_artist_sql(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        cache_key = f"artist_{hash(user_input.lower())}"
        
        def _generate():
            artist_name = self._extract_entity(user_input, "artist")
            
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
                metadata = {"query_type": "artist_info_no_match", "error": "No specific artist found"}
            
            return sql, metadata
        
        return self._get_cached_or_generate(cache_key, _generate)
    
    def generate_recommendation_sql(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        cache_key = f"recommend_{hash(user_input.lower())}"
        
        def _generate():
            artist_name = self._extract_entity(user_input, "similarity")
            
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
                metadata = {"query_type": "recommend_no_match", "error": "No target artist found for recommendations"}
            
            return sql, metadata
        
        return self._get_cached_or_generate(cache_key, _generate)
    
    def generate_stats_sql(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        cache_key = f"stats_{hash(user_input.lower())}"
        
        def _generate():
            content_type = self._extract_content_type(user_input)
            time_period = self._extract_time_period(user_input)
            
            if content_type == "tracks":
                sql = f"""
                    SELECT t.name, ar.name as artist_name, utt.playcount 
                    FROM user_top_tracks utt 
                    JOIN tracks t ON utt.track_id = t.track_id 
                    JOIN artists ar ON t.artist_id = ar.artist_id
                    WHERE utt.time_period = '{time_period}'
                    ORDER BY utt.playcount DESC LIMIT 10
                """
            elif content_type == "albums":
                sql = f"""
                    SELECT a.name, ar.name as artist_name, uta.playcount 
                    FROM user_top_albums uta 
                    JOIN albums a ON uta.album_id = a.album_id 
                    JOIN artists ar ON a.artist_id = ar.artist_id
                    WHERE uta.time_period = '{time_period}'
                    ORDER BY uta.playcount DESC LIMIT 10
                """
            else:  # artists
                sql = f"""
                    SELECT a.name, uta.playcount 
                    FROM user_top_artists uta 
                    JOIN artists a ON uta.artist_id = a.artist_id 
                    WHERE uta.time_period = '{time_period}'
                    ORDER BY uta.playcount DESC LIMIT 10
                """
            
            metadata = {"query_type": f"listening_stats_{content_type}_{time_period}"}
            return sql, metadata
        
        return self._get_cached_or_generate(cache_key, _generate)
    
    def _extract_entity(self, text: str, extraction_type: str) -> str:
        text = text.lower()
        patterns = ARTIST_PATTERNS if extraction_type == "artist" else SIMILARITY_PATTERNS
        cleanup_pattern = CLEANUP_WORDS[extraction_type]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                candidate = re.sub(cleanup_pattern, '', candidate, flags=re.IGNORECASE).strip()
                if candidate and len(candidate) > 1:
                    return candidate
        
        return None
    
    def _extract_content_type(self, text: str) -> str:
        text = text.lower()
        for content_type, keywords in CONTENT_TYPE_KEYWORDS.items():
            if any(word in text for word in keywords):
                return content_type
        return "artists"
    
    def _extract_time_period(self, text: str) -> str:
        text = text.lower()
        for time_period in TIME_PERIODS:
            if time_period in text:
                return time_period
        return "overall"
    
    def get_cache_stats(self) -> Dict[str, int]:
        return {"total_cache_entries": len(self.cache), "total_queries": len(self.query_history)}


class ResponseGenerator:
    def __init__(self, text_generator: SimpleTextGenerator):
        self.text_generator = text_generator
        self.response_history = []
        self.user_preferences = {}
    
    def generate_response(self, state: AgentState) -> str:
        if not state["results"]:
            return self._handle_empty_results(state)
        
        unique_results = self._remove_duplicates(state["results"])
        formatted_results = self._format_results(unique_results, state["intent"])
        
        completion_prompt = self._create_completion_prompt(
            state["intent"], 
            formatted_results, 
            state["user_input"]
        )
        
        response = self._try_llm_generation(completion_prompt)
        
        if not response:
            response = self._template_fallback(state["intent"], formatted_results)
        
        self._track_response(state, response)
        return response
    
    def _create_completion_prompt(self, intent: str, formatted_results: str, user_input: str) -> str:
        if intent == "artist_info":
            artist_name = formatted_results.split(':')[0].split('(')[0].strip()
            
            if ':' in formatted_results:
                return f"I found {artist_name} in your music library. {artist_name} is"
            else:
                return f"I found {artist_name} with music in your collection. This artist is known for"
        
        elif intent == "listening_stats":
            if "7day" in user_input or "week" in user_input:
                return f"Looking at your recent listening, your top music is {formatted_results}. This week you've been enjoying"
            elif "overall" in user_input:
                return f"Your all-time favorites include {formatted_results}. Your music taste shows you really love"
            else:
                return f"Your music listening shows {formatted_results}. This indicates that you tend to enjoy"
        
        elif intent == "recommend_music":
            if "similarity" in formatted_results:
                return f"Based on your taste, I recommend {formatted_results}. These artists are worth exploring because they"
            else:
                return f"I suggest checking out {formatted_results}. You might enjoy them since they"
        
        return f"Looking at your music data: {formatted_results}. This shows"
    
    def _template_fallback(self, intent: str, formatted_results: str) -> str:
        templates = {
            "artist_info": f"I found {formatted_results} in your music library!",
            "listening_stats": f"Here's what you've been listening to: {formatted_results}",
            "recommend_music": f"I recommend: {formatted_results}"
        }
        return templates.get(intent, f"Results: {formatted_results}")

    def _try_llm_generation(self, prompt: str) -> str:
        try:
            response = self.text_generator.invoke(prompt)
            if response and len(response.strip()) >= 10:
                return response.strip()
        except Exception as e:
            print(f"LLM generation failed: {e}")
        return None
    
    def _handle_empty_results(self, state: AgentState) -> str:
        metadata = state.get("query_metadata", {})
        error = metadata.get("error", "")
        
        if "no specific artist found" in error.lower():
            return "I couldn't identify a specific artist. Please specify an artist name."
        
        if "no target artist found" in error.lower():
            return "Please specify which artist you want recommendations for."
        
        if (state["intent"] == "recommend_music" and 
            metadata.get("query_type", "").startswith("recommend_similar_")):
            artist = metadata.get("query_type", "").replace("recommend_similar_", "")
            return f"No similarity data found for {artist} in your library."
        
        return {
            "artist_info": "I couldn't find that artist in your library.",
            "listening_stats": "I couldn't find listening data for that request.",
            "recommend_music": "Please specify an artist for recommendations."
        }.get(state["intent"], "I couldn't understand your request.")
    
    def _remove_duplicates(self, results: List[Dict]) -> List[Dict]:
        seen, unique = set(), []
        for result in results:
            name = result.get('name', 'Unknown')
            if name not in seen:
                seen.add(name)
                unique.append(result)
        return unique
    
    def _format_results(self, results: List[Dict], intent: str) -> str:
        if intent == "artist_info":
            result = results[0]
            name = result.get('name', 'Unknown')
            bio = result.get('bio_summary', '').strip()
            playcount = result.get('playcount', 0)
            
            if bio:
                bio_text = bio[:100] + "..." if len(bio) > 100 else bio
                return f"{name}: {bio_text}" + (f" ({playcount:,} plays)" if playcount else "")
            return f"{name}" + (f" ({playcount:,} plays)" if playcount else "")
        
        formatted = []
        for result in results[:3]:
            if 'similar_artist' in result:
                name = result.get('similar_artist', 'Unknown')
                score = result.get('match_score', 0)
                formatted.append(f"{name} (similarity: {score:.2f})")
            else:
                name = result.get('name', 'Unknown')
                artist_name = result.get('artist_name', '')
                count = result.get('playcount', 0)
                
                display_name = f"{name} by {artist_name}" if artist_name else name
                formatted.append(f"{display_name} ({count:,} plays)" if count else display_name)
        
        return ", ".join(formatted)
    
    def _track_response(self, state: AgentState, response: str) -> None:
        self.response_history.append({
            "user_input": state["user_input"],
            "intent": state["intent"],
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        intent = state["intent"]
        if intent not in self.user_preferences:
            self.user_preferences[intent] = {"count": 0}
        self.user_preferences[intent]["count"] += 1
    
    def get_user_preferences(self) -> Dict[str, Any]:
        return self.user_preferences

# =============================================================================
# SETUP FUNCTION  
# =============================================================================

def setup_llm() -> SimpleTextGenerator:
    """Setup only the text generation model (no classification model needed)"""
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers not available")
        raise ImportError("Transformers library is required")
    
    try:
        print("Loading text generation model...")
        generator = pipeline("text-generation", model="distilgpt2", max_length=150, pad_token_id=50256)
        generation_llm = SimpleTextGenerator(generator)
        print("Text generation model loaded successfully")
        
        return generation_llm
    except Exception as e:
        print(f"Model loading failed: {e}")
        raise RuntimeError(f"Failed to load text generation model: {e}")


# =============================================================================
# MAIN AGENT CLASS
# =============================================================================

class MusicDiscoveryAgent:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.classifier = PatternBasedClassifier()
        self.generation_llm = setup_llm()
        self.sql_generator = SQLGenerator()
        self.response_generator = ResponseGenerator(self.generation_llm)
        self.session_history = []
        self.graph = self._create_graph()
    
    def _create_graph(self) -> Any:
        graph = StateGraph(AgentState)
        
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("get_bio_info", self._generate_artist_sql)
        graph.add_node("get_recommendation", self._generate_recommendation_sql)
        graph.add_node("describe_my_listening", self._generate_stats_sql)
        graph.add_node("execute_sql_query", self._execute_query)
        graph.add_node("generate_response", self._generate_response)
        
        graph.add_edge(START, "classify_intent")
        graph.add_conditional_edges(
            "classify_intent", self._route_sql_type,
            {"get_bio_info": "get_bio_info", "get_recommendation": "get_recommendation", "describe_my_listening": "describe_my_listening"}
        )
        graph.add_edge("get_bio_info", "execute_sql_query")
        graph.add_edge("get_recommendation", "execute_sql_query")
        graph.add_edge("describe_my_listening", "execute_sql_query")
        graph.add_edge("execute_sql_query", "generate_response")
        graph.add_edge("generate_response", END)
        
        return graph.compile()
    
    def _classify_intent(self, state: AgentState) -> AgentState:
        try:
            state["intent"] = self.classifier.classify(state["user_input"])
        except Exception as e:
            print(f"Intent classification error: {e}")
            state["intent"] = "recommend_music"
        return state
    
    def _route_sql_type(self, state: AgentState) -> str:
        routing = {"artist_info": "get_bio_info", "listening_stats": "describe_my_listening"}
        return routing.get(state["intent"], "get_recommendation")
    
    def _generate_artist_sql(self, state: AgentState) -> AgentState:
        sql, metadata = self.sql_generator.generate_artist_sql(state["user_input"])
        state["sql_query"] = sql
        state["query_metadata"] = metadata
        return state
    
    def _generate_recommendation_sql(self, state: AgentState) -> AgentState:
        sql, metadata = self.sql_generator.generate_recommendation_sql(state["user_input"])
        state["sql_query"] = sql
        state["query_metadata"] = metadata
        return state
    
    def _generate_stats_sql(self, state: AgentState) -> AgentState:
        sql, metadata = self.sql_generator.generate_stats_sql(state["user_input"])
        state["sql_query"] = sql
        state["query_metadata"] = metadata
        return state
    
    def _execute_query(self, state: AgentState) -> AgentState:
        try:
            sql = state["sql_query"]
            if sql and sql.strip():
                results = self.db_manager.execute_query(sql)
                state["results"] = results[:10]
            else:
                state["results"] = []
                print(f"  No SQL generated - couldn't understand request")
        except Exception as e:
            print(f"Database error: {e}")
            state["results"] = []
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        try:
            state["response"] = self.response_generator.generate_response(state)
        except Exception as e:
            print(f"Response generation error: {e}")
            state["response"] = "I found some music information for you!"
        return state
    
    def process_query(self, user_input: str) -> Dict[str, Any]:
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
            print(f"Error processing query: {e}")
            return {**state, "response": "Sorry, I encountered an error processing your request."}
    
    def get_session_stats(self) -> Dict[str, Any]:
        intent_dist = {}
        for entry in self.session_history:
            if "intent" in entry:
                intent = entry["intent"]
                intent_dist[intent] = intent_dist.get(intent, 0) + 1
        
        return {
            "total_queries": len(self.session_history),
            "sql_cache_stats": self.sql_generator.get_cache_stats(),
            "user_preferences": self.response_generator.get_user_preferences(),
            "intent_distribution": intent_dist
        }
    
    def clear_session(self) -> None:
        self.session_history.clear()
        if hasattr(self.sql_generator, 'cache'):
            self.sql_generator.cache.clear()
        if hasattr(self.response_generator, 'response_history'):
            self.response_generator.response_history.clear()
        print("Session cleared")


# =============================================================================
# PUBLIC INTERFACES
# =============================================================================

def create_music_agent(db_path: str) -> MusicDiscoveryAgent:
    """Create a music discovery agent with database connection."""
    return MusicDiscoveryAgent(DatabaseManager(db_path))