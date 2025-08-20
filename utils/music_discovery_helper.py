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

INTENT_LABELS = [
    "tell me biographical information about a specific artist or band",
    "recommend new music for me to discover", 
    "show me MY personal listening statistics and data"
]

STATS_KEYWORDS = ["my top", "my most", "my listening", "what did i listen"]
TIME_PATTERN = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|last month|this month|last year|this year|in 20\d{2})\b'

GENRE_PATTERNS = {
    'chill': ['relax', 'calm', 'chill', 'ambient'],
    'jazz': ['jazz', 'blues'],
    'rock': ['rock', 'metal'],
    'electronic': ['electronic', 'techno', 'edm']
}

ARTIST_PATTERNS = [
    r'tell me about (.+)', r'who is (.+)\?', r'about (.+)', r'information about (.+)'
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
        
        # Check for statistics keywords first
        if (any(keyword in message.lower() for keyword in STATS_KEYWORDS) or
            re.search(TIME_PATTERN, message, re.IGNORECASE)):
            print("  🎯 Keyword override: detected statistics request")
            return "listening_stats"
        
        # Use zero-shot classification
        result = self.classifier(message, INTENT_LABELS)
        
        # Debug output
        print(f"  🔍 Message: '{message}'")
        for i, (label, score) in enumerate(zip(result['labels'], result['scores']), 1):
            print(f"  🔍 Label {i}: '{label}' (score: {score:.3f})")
        
        # Map result to intent
        top_label = result['labels'][0]
        if "biographical information about" in top_label:
            return "artist_info"
        elif "MY personal listening" in top_label:
            return "listening_stats"
        else:
            return "recommend_music"


class SimpleLLM:
    def invoke(self, prompt_text: str) -> str:
        if "classify this music request" not in prompt_text.lower():
            return "Fallback classifier: Unable to process this request type."
            
        message = prompt_text.split("Request: ")[-1].split("\n")[0].lower()
        
        stats_pattern = r'\b(my top|statistics|listening habits|most played|january|february|march|april|may|june|july|august|september|october|november|december|last month|this month|last year|this year|in 20\d{2})\b'
        if re.search(stats_pattern, message):
            return "listening_stats"
        
        if any(word in message for word in ["tell me about", "who is", "what is", "info"]):
            return "artist_info"
        
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
            print(f"  💾 Using cached query: {metadata.get('query_type', 'unknown')}")
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
            
            for pattern in ARTIST_PATTERNS:
                match = re.search(pattern, message)
                if match:
                    artist_name = match.group(1).strip()
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
            detected_genre = "general"
            
            for genre, keywords in GENRE_PATTERNS.items():
                if any(word in message for word in keywords):
                    detected_genre = genre
                    sql = f"SELECT a.name, a.playcount FROM artists a JOIN artist_tags at ON a.artist_id = at.artist_id JOIN tags t ON at.tag_id = t.tag_id WHERE t.name LIKE '%{genre}%' ORDER BY a.playcount DESC LIMIT 5"
                    break
            else:
                sql = "SELECT name, playcount FROM artists ORDER BY playcount DESC LIMIT 5"
            
            metadata = {
                "detected_genre": detected_genre,
                "query_type": "recommendation",
                "timestamp": datetime.now().isoformat()
            }
            return sql, metadata
        
        return self._get_cached_or_generate(cache_key, _generate)
    
    def generate_stats_sql(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        cache_key = f"stats_{hash(user_input.lower())}"
        
        def _generate():
            message = user_input.lower()
            
            if "top" in message and "artist" in message:
                sql = "SELECT a.name, uta.playcount FROM user_top_artists uta JOIN artists a ON uta.artist_id = a.artist_id ORDER BY uta.playcount DESC LIMIT 10"
                stats_type = "top_artists"
            elif "top" in message and ("track" in message or "song" in message):
                sql = "SELECT t.name, utt.playcount FROM user_top_tracks utt JOIN tracks t ON utt.track_id = t.track_id ORDER BY utt.playcount DESC LIMIT 10"
                stats_type = "top_tracks"
            elif ("most played" in message and "song" in message) or ("played song" in message):
                sql = "SELECT t.name, utt.playcount FROM user_top_tracks utt JOIN tracks t ON utt.track_id = t.track_id ORDER BY utt.playcount DESC LIMIT 10"
                stats_type = "most_played_songs"
            elif "march" in message:
                sql = "SELECT a.name, COUNT(*) as listens FROM user_listening_history ulh JOIN tracks t ON ulh.track_id = t.track_id JOIN artists a ON t.artist_id = a.artist_id WHERE strftime('%m', ulh.listened_at) = '03' GROUP BY a.artist_id ORDER BY listens DESC LIMIT 10"
                stats_type = "march_listening"
            else:
                sql = "SELECT a.name, uta.playcount FROM user_top_artists uta JOIN artists a ON uta.artist_id = a.artist_id ORDER BY uta.playcount DESC LIMIT 5"
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
    
    def generate_response(self, state: AgentState) -> str:
        if not state["results"]:
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
    
    def _get_empty_response(self, intent: str) -> str:
        responses = {
            "artist_info": "I couldn't find information about that artist in your listening history.",
            "listening_stats": "I couldn't find any listening statistics for that request.",
        }
        return responses.get(intent, "I couldn't find any music matching your request.")
    
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
            playcount = result.get('playcount', 0)
            return f"{name} with {playcount:,} plays" if playcount else name
        
        formatted = []
        for result in results[:3]:
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
        
        if intent == "recommend_music" and "detected_genre" in query_metadata:
            genre = query_metadata["detected_genre"]
            patterns = self.user_preferences[intent]["patterns"]
            patterns[genre] = patterns.get(genre, 0) + 1
    
    def get_user_preferences(self) -> Dict[str, Any]:
        return self.user_preferences


# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def setup_llm() -> Tuple[Any, SimpleTextGenerator]:
    if not TRANSFORMERS_AVAILABLE:
        print("⚠️ Transformers not available, using rule-based fallback")
        return SimpleLLM(), SimpleLLM()
    
    try:
        print("Loading models...")
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        classification_llm = ZeroShotMusicClassifier(classifier)
        
        generator = pipeline("text-generation", model="distilgpt2", max_length=150, pad_token_id=50256)
        generation_llm = SimpleTextGenerator(generator)
        print("✅ Models loaded successfully")
        
        return classification_llm, generation_llm
    except Exception as e:
        print(f"⚠️ Model loading failed: {e}. Using fallback.")
        return SimpleLLM(), SimpleLLM()


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
        
        # Add nodes
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("artist_sql", self._generate_artist_sql)
        graph.add_node("recommendation_sql", self._generate_recommendation_sql)
        graph.add_node("stats_sql", self._generate_stats_sql)
        graph.add_node("execute_query", self._execute_query)
        graph.add_node("generate_response", self._generate_response)
        
        # Add edges
        graph.add_edge(START, "classify_intent")
        graph.add_conditional_edges(
            "classify_intent", self._route_sql_type,
            {"artist_sql": "artist_sql", "recommendation_sql": "recommendation_sql", "stats_sql": "stats_sql"}
        )
        graph.add_edge("artist_sql", "execute_query")
        graph.add_edge("recommendation_sql", "execute_query")
        graph.add_edge("stats_sql", "execute_query")
        graph.add_edge("execute_query", "generate_response")
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
        routing = {"artist_info": "artist_sql", "listening_stats": "stats_sql"}
        return routing.get(state["intent"], "recommendation_sql")
    
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
        print("🧹 Session cleared")


# =============================================================================
# PUBLIC INTERFACES
# =============================================================================

def create_music_agent(db_path: str) -> MusicDiscoveryAgent:
    return MusicDiscoveryAgent(DatabaseManager(db_path))


def query_music_agent(user_input: str, db_manager: DatabaseManager) -> Dict[str, Any]:
    agent = MusicDiscoveryAgent(db_manager)
    return agent.process_query(user_input)


def run_music_chatbot(db_path: str) -> None:
    agent = create_music_agent(db_path)
    print("🎵 Music Discovery AI Ready! Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break
        elif user_input.lower() == 'stats':
            stats = agent.get_session_stats()
            print(f"📊 Queries: {stats['total_queries']}, Cache: {stats['sql_cache_stats']}")
            continue
        elif user_input.lower() == 'clear':
            agent.clear_session()
            continue
        
        result = agent.process_query(user_input)
        print(f"🎵 {result['response']}")


def test_agent(db_path: str) -> None:
    agent = create_music_agent(db_path)
    
    test_queries = [
        "Recommend some relaxing music", "Tell me about Radiohead", "Show me my top 10 artists",
        "Who is Taylor Swift?", "Find me some jazz music", "What artist did I listen to most in March?",
        "Show me my most played songs", "Recommend some relaxing music"  # Test caching
    ]
    
    for query in test_queries:
        print(f"\nTesting: {query}")
        result = agent.process_query(query)
        print(f"Intent: {result['intent']}")
        print(f"Response: {result['response']}")
    
    stats = agent.get_session_stats()
    print(f"\n📊 Final Stats: {stats['total_queries']} queries, {stats['sql_cache_stats']} cache stats")