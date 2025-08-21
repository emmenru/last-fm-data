# utils/music_discovery_helper.py
"""
Stateful Music Discovery AI Agent Helper
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

# Improved intent labels for better zero-shot classification
INTENT_LABELS = [
    "provide biographical information, background details, or career information about a specific named artist, band, or musician",
    "recommend new music to discover, suggest similar artists to explore, find music recommendations, or get music suggestions", 
    "display my existing personal listening statistics, show my top played artists or songs from my listening history"
]


# =============================================================================
# CORE CLASSES
# =============================================================================

class AgentState(TypedDict):
    user_input: str
    intent: str
    sql_query: str
    results: List[Dict]
    response: str
    db_manager: Any
    agent: Any
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


class ZeroShotMusicClassifier:
    def __init__(self, classifier):
        self.classifier = classifier
    
    def invoke(self, prompt_text: str) -> str:
        if "classify this music request" not in prompt_text.lower():
            return "Zero-shot classifier: Unable to process this request type."
            
        message = prompt_text.split("Request: ")[-1].split("\n")[0]
        query_lower = message.lower()
        
        # Pattern-based classification with compact lookup
        patterns = {
            "listening_stats": ["my top", "my most", "show me my", "my listening", "my statistics", "my personal", "my history"],
            "artist_info": ["tell me about", "who is", "information about", "biography for"],
            "recommend_music": ["recommend", "suggest", "discover", "find artists", "similar to", "music like", "artists like"]
        }
        
        # Check patterns with special handling for artist_info (needs artist keywords)
        for intent, phrases in patterns.items():
            if any(phrase in query_lower for phrase in phrases):
                if intent == "artist_info":
                    # Artist info needs additional artist-related keywords
                    if any(word in query_lower for word in ["artist", "band", "musician"]):
                        print(f"  Pattern override: detected {intent.replace('_', ' ')} request")
                        return intent
                else:
                    print(f"  Pattern override: detected {intent.replace('_', ' ')} request")
                    return intent
        
        # Use zero-shot classification for other cases
        result = self.classifier(message, INTENT_LABELS)
        
        # Debug output
        print(f"  Message: '{message}'")
        for i, (label, score) in enumerate(zip(result['labels'], result['scores']), 1):
            print(f"  Label {i}: '{label[:50]}...' (score: {score:.3f})")
        
        # Map result to intent based on label content
        top_label = result['labels'][0].lower()
        
        if "biographical" in top_label or ("information" in top_label and "specific" in top_label):
            return "artist_info"
        elif "existing" in top_label or ("display my" in top_label and "listening" in top_label):
            return "listening_stats"
        elif "recommend" in top_label or "suggest" in top_label or "discover" in top_label or "explore" in top_label:
            return "recommend_music"
        else:
            # Final fallback - default to recommendation for ambiguous cases
            return "recommend_music"


class SimpleTextGenerator:
    def __init__(self, generator):
        self.generator = generator
        self.bad_patterns = ["Give a friendly response:", "Response:", "Friendly summary:", "Friendly recommendation:"]
    
    def invoke(self, prompt_text: str) -> str:
        try:
            responses = self.generator(
                prompt_text, max_new_tokens=30, do_sample=True, temperature=0.5,
                top_p=0.9, top_k=50, return_full_text=False, clean_up_tokenization_spaces=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            
            if not responses:
                return "Here are your music results!"
            
            response = responses[0]['generated_text'].strip().split('\n')[0]
            
            # Clean response
            for pattern in self.bad_patterns:
                response = response.replace(pattern, "")
            response = response.strip()
            
            # Validate quality
            if (len(response) < 5 or 
                not any(word in response.lower() for word in ['music', 'artist', 'song', 'found', 'here'])):
                return "Here are your music results!"
            
            return response
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "Here are your music results!"


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
        self.cache[cache_key] = (sql, metadata)
        self.query_history.append({"sql": sql, "metadata": metadata})
        return sql, metadata
    
    def generate_artist_sql(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        cache_key = f"artist_{hash(user_input.lower())}"
        
        def _generate():
            message = user_input.lower()
            artist_name = None
            
            # More flexible artist name extraction
            patterns = [
                r'(?:tell me about|about|who is|information about)\s+(.+?)(?:\?|$)',
                r'(?:artist|band|musician)\s+(.+?)(?:\?|$)',
                r'(.+?)(?:\s+(?:artist|band|musician))?(?:\?|$)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    # Filter out common stop words
                    if candidate and candidate not in ['the', 'a', 'an', 'is', 'are', 'was', 'were']:
                        artist_name = candidate
                        break
            
            if artist_name:
                sql = f"SELECT name, bio_summary, listeners, playcount FROM artists WHERE name LIKE '%{artist_name}%' LIMIT 5"
                metadata = {"artist_name": artist_name, "query_type": "artist_info"}
            else:
                sql = "SELECT name, bio_summary, listeners, playcount FROM artists ORDER BY playcount DESC LIMIT 5"
                metadata = {"artist_name": "top_artists", "query_type": "artist_info_default"}
            
            metadata["timestamp"] = datetime.now().isoformat()
            return sql, metadata
        
        return self._get_cached_or_generate(cache_key, _generate)
    
    def generate_recommendation_sql(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        cache_key = f"recommend_{hash(user_input.lower())}"
        
        def _generate():
            message = user_input.lower()
            
            # Extract artist name from recommendation request
            artist_name = None
            patterns = [
                r'(?:similar to|like|artists like|music like|bands like)\s+(.+?)(?:\?|$)',
                r'(?:find|recommend|suggest).*(?:similar.*to|like)\s+(.+?)(?:\?|$)',
                r'(?:based on|inspired by)\s+(.+?)(?:\?|$)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    # Clean up common words
                    candidate = re.sub(r'\b(the artist|artist|band|musician)\b', '', candidate, flags=re.IGNORECASE).strip()
                    if candidate:
                        artist_name = candidate
                        break
            
            if artist_name:
                # Use similarity table to find similar artists
                sql = (f"SELECT a2.name as similar_artist, ars.match_score "
                       f"FROM artists a1 "
                       f"JOIN artist_similar ars ON a1.artist_id = ars.artist_id "
                       f"JOIN artists a2 ON ars.similar_artist_id = a2.artist_id "
                       f"WHERE a1.name LIKE '%{artist_name}%' "
                       f"ORDER BY ars.match_score DESC LIMIT 10")
                metadata = {
                    "query_type": "similarity_recommendation",
                    "target_artist": artist_name,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Fallback to popular artists if no specific artist mentioned
                sql = "SELECT name, playcount FROM artists ORDER BY playcount DESC LIMIT 10"
                metadata = {
                    "query_type": "popular_recommendation",
                    "timestamp": datetime.now().isoformat()
                }
            
            return sql, metadata
        
        return self._get_cached_or_generate(cache_key, _generate)
    
    def generate_stats_sql(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        cache_key = f"stats_{hash(user_input.lower())}"
        
        def _generate():
            message = user_input.lower()
            
            # Enhanced pattern matching for statistics
            if any(word in message for word in ["top", "most played", "favorite"]) and any(word in message for word in ["artist", "artists"]):
                sql = "SELECT a.name, uta.playcount FROM user_top_artists uta JOIN artists a ON uta.artist_id = a.artist_id ORDER BY uta.playcount DESC LIMIT 10"
                stats_type = "top_artists"
            elif any(word in message for word in ["top", "most played", "favorite"]) and any(word in message for word in ["track", "song", "songs", "tracks"]):
                sql = "SELECT t.name, utt.playcount FROM user_top_tracks utt JOIN tracks t ON utt.track_id = t.track_id ORDER BY utt.playcount DESC LIMIT 10"
                stats_type = "top_tracks"
            elif any(month in message for month in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]):
                # Extract month number for SQL query
                month_map = {"january": "01", "february": "02", "march": "03", "april": "04", 
                           "may": "05", "june": "06", "july": "07", "august": "08",
                           "september": "09", "october": "10", "november": "11", "december": "12"}
                month_num = "03"  # Default to March as example
                for month_name, num in month_map.items():
                    if month_name in message:
                        month_num = num
                        break
                sql = f"SELECT a.name, COUNT(*) as listens FROM user_listening_history ulh JOIN tracks t ON ulh.track_id = t.track_id JOIN artists a ON t.artist_id = a.artist_id WHERE strftime('%m', ulh.listened_at) = '{month_num}' GROUP BY a.artist_id ORDER BY listens DESC LIMIT 10"
                stats_type = f"monthly_listening_{month_num}"
            elif any(word in message for word in ["listening", "history", "played", "statistics", "stats"]):
                sql = "SELECT a.name, uta.playcount FROM user_top_artists uta JOIN artists a ON uta.artist_id = a.artist_id ORDER BY uta.playcount DESC LIMIT 10"
                stats_type = "general_stats"
            else:
                sql = "SELECT a.name, uta.playcount FROM user_top_artists uta JOIN artists a ON uta.artist_id = a.artist_id ORDER BY uta.playcount DESC LIMIT 10"
                stats_type = "default_top_artists"
            
            metadata = {
                "stats_type": stats_type,
                "query_type": "listening_stats",
                "timestamp": datetime.now().isoformat()
            }
            return sql, metadata
        
        return self._get_cached_or_generate(cache_key, _generate)
    
    def get_cache_stats(self) -> Dict[str, int]:
        return {"total_cache_entries": len(self.cache), "total_queries": len(self.query_history)}


class ResponseGenerator:
    def __init__(self, text_generator: SimpleTextGenerator):
        self.text_generator = text_generator
        self.response_history = []
        self.user_preferences = {}
    
    def _get_empty_response(self, intent: str) -> str:
        responses = {
            "artist_info": "I couldn't find information about that artist in your listening history.",
            "listening_stats": "I couldn't find any listening statistics for that request.",
        }
        return responses.get(intent, "I couldn't find any music matching your request.")
    
    def generate_response(self, state: AgentState) -> str:
        if not state["results"]:
            # Special handling for similarity requests with no results
            if (state["intent"] == "recommend_music" and 
                state.get("query_metadata", {}).get("query_type") == "similarity_recommendation"):
                target_artist = state.get("query_metadata", {}).get("target_artist", "that artist")
                print(f"  No similarity data found for '{target_artist}' in local database")
                print(f"  Would need to search Last.fm API for similar artists to '{target_artist}'")
                return f"No similarity data found for {target_artist} in your local library. This would require searching the Last.fm API for similar artists."
            else:
                response = self._get_empty_response(state["intent"])
        else:
            unique_results = self._remove_duplicates(state["results"])
            results_text = self._format_results(unique_results, state["intent"])
            prompt = self._create_prompt(state["intent"], results_text)
            response = self.text_generator.invoke(prompt)
            response = self._validate_response(response, state["intent"], results_text)
        
        # Track response
        self.response_history.append({
            "user_input": state["user_input"],
            "intent": state["intent"],
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update preferences
        self._update_preferences(state["intent"], state.get("query_metadata", {}))
        return response
    
    def _remove_duplicates(self, results: List[Dict]) -> List[Dict]:
        seen_names = set()
        unique_results = []
        for result in results:
            name = result.get('name', 'Unknown')
            if name not in seen_names:
                seen_names.add(name)
                unique_results.append(result)
        return unique_results
    
    def _format_results(self, results: List[Dict], intent: str) -> str:
        if intent == "artist_info":
            result = results[0]
            name = result.get('name', 'Unknown')
            bio_summary = result.get('bio_summary', '')
            playcount = result.get('playcount', 0)
            
            # Include bio information if available
            if bio_summary and len(bio_summary.strip()) > 0:
                # Truncate bio if too long for response generation
                bio_text = bio_summary[:200] + "..." if len(bio_summary) > 200 else bio_summary
                return f"{name}: {bio_text} ({playcount:,} plays)" if playcount else f"{name}: {bio_text}"
            else:
                return f"{name} with {playcount:,} plays" if playcount else name
        
        # For recommendations and stats, format differently
        formatted = []
        for result in results[:3]:
            # Handle similarity results (have 'similar_artist' field)
            if 'similar_artist' in result:
                name = result.get('similar_artist', 'Unknown')
                score = result.get('match_score', 0)
                formatted.append(f"{name} (similarity: {score:.2f})" if score else name)
            else:
                # Regular results
                name = result.get('name', 'Unknown')
                count = result.get('playcount', result.get('listens', 0))
                formatted.append(f"{name} ({count:,} plays)" if count else name)
        return ", ".join(formatted)
    
    def _create_prompt(self, intent: str, results_text: str) -> str:
        prompts = {
            "artist_info": f"I found the artist {results_text}. Here's what I can tell you:",
            "listening_stats": f"Your music stats show {results_text}. Summary:",
        }
        return prompts.get(intent, f"I recommend these artists: {results_text}. Great choices:")
    
    def _validate_response(self, response: str, intent: str, results_text: str) -> str:
        if (not response or len(response) < 10 or response == "Here are your music results!" or
            "number of times" in response.lower() or response.endswith(",")):
            
            fallbacks = {
                "artist_info": f"I found {results_text} in your music library!",
                "listening_stats": f"Here are your top results: {results_text}",
            }
            return fallbacks.get(intent, f"I recommend these artists: {results_text}")
        return response
    
    def _update_preferences(self, intent: str, query_metadata: Dict[str, Any]) -> None:
        if intent not in self.user_preferences:
            self.user_preferences[intent] = {"count": 0, "patterns": {}}
        
        self.user_preferences[intent]["count"] += 1
    
    def get_user_preferences(self) -> Dict[str, Any]:
        return self.user_preferences


# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def setup_llm() -> Tuple[Any, SimpleTextGenerator]:
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers not available")
        raise ImportError("Transformers library is required")
    
    try:
        print("Loading models...")
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        classification_llm = ZeroShotMusicClassifier(classifier)
        
        generator = pipeline("text-generation", model="distilgpt2", max_length=150, pad_token_id=50256)
        generation_llm = SimpleTextGenerator(generator)
        print("Models loaded successfully")
        
        return classification_llm, generation_llm
    except Exception as e:
        print(f"Model loading failed: {e}")
        raise RuntimeError(f"Failed to load models: {e}")


# =============================================================================
# MAIN AGENT CLASS
# =============================================================================

class MusicDiscoveryAgent:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.classification_llm, self.generation_llm = setup_llm()
        self.sql_generator = SQLGenerator()
        self.response_generator = ResponseGenerator(self.generation_llm)
        self.session_history = []
        self.graph = self._create_graph()
    
    def _create_graph(self) -> Any:
        graph = StateGraph(AgentState)
        
        # Add nodes with more descriptive names
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("get_bio_info", self._generate_artist_sql)
        graph.add_node("get_recommendation", self._generate_recommendation_sql)
        graph.add_node("describe_my_listening", self._generate_stats_sql)
        graph.add_node("execute_sql_query", self._execute_query)
        graph.add_node("generate_response", self._generate_response)
        
        # Add edges with updated node names
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
            prompt = f"Classify this music request into one category: recommend_mood, artist_info, top_music, or general_search\nRequest: {state['user_input']}\nCategory:"
            state["intent"] = self.classification_llm.invoke(prompt).strip()
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
            results = self.db_manager.execute_query(state["sql_query"])
            state["results"] = results[:10]
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
            "user_input": user_input, "intent": "", "sql_query": "", "results": [],
            "response": "", "db_manager": self.db_manager, "agent": self, "query_metadata": {}
        }
        
        try:
            final_state = self.graph.invoke(state)
            self.session_history.append({
                "user_input": user_input, "intent": final_state["intent"],
                "response": final_state["response"], "timestamp": datetime.now().isoformat()
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
        self.session_history = []
        self.sql_generator = SQLGenerator()
        self.response_generator = ResponseGenerator(self.generation_llm)
        print("Session cleared")


# =============================================================================
# PUBLIC INTERFACES
# =============================================================================

def create_music_agent(db_path: str) -> MusicDiscoveryAgent:
    """Create a music discovery agent with database connection."""
    return MusicDiscoveryAgent(DatabaseManager(db_path))


def test_agent(db_path: str) -> None:
    """Test the agent with the three main intent cases."""
    agent = create_music_agent(db_path)
    
    # Test the three main intents
    test_queries = [
        "Tell me about Radiohead",                    # artist_info
        "Find artists similar to Radiohead",         # recommend_music  
        "Show me my top 10 artists"                  # listening_stats
    ]
    
    print("Testing three main intents:")
    for query in test_queries:
        print(f"\nTesting: {query}")
        result = agent.process_query(query)
        print(f"  Intent: {result['intent']}")
        print(f"  SQL: {result['sql_query']}")
        print(f"  Results: {len(result['results'])} found")
        print(f"  Response: {result['response']}")
    
    # Show final stats
    stats = agent.get_session_stats()
    print(f"\nTest completed: {stats['total_queries']} queries processed")
    print(f"Cache efficiency: {stats['sql_cache_stats']}")
    print(f"Intent distribution: {stats['intent_distribution']}")