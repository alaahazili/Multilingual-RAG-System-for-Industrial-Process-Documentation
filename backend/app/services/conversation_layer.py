"""
Conversation & Memory Layer
Purpose: Enable chatting with one document (not just independent Q&A).
Store conversation history (last 3–5 interactions).
Pass history back to the model with each new query.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single turn in a conversation"""
    timestamp: str
    user_query: str
    bot_response: str
    retrieved_contexts: List[str]
    search_scores: List[float]
    generation_time: float
    search_time: float


class ConversationMemory:
    """Manages conversation history for RAG system"""
    
    def __init__(self, max_history: int = 5):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of conversation turns to remember
        """
        self.max_history = max_history
        self.conversation_history: deque = deque(maxlen=max_history)
        self.session_id: Optional[str] = None
        
    def add_turn(self, turn: ConversationTurn):
        """Add a new conversation turn to history"""
        self.conversation_history.append(turn)
        logger.info(f"Added conversation turn. History size: {len(self.conversation_history)}")
        
    def get_recent_history(self, turns: int = 3) -> List[ConversationTurn]:
        """Get recent conversation history"""
        return list(self.conversation_history)[-turns:]
        
    def get_full_history(self) -> List[ConversationTurn]:
        """Get full conversation history"""
        return list(self.conversation_history)
        
    def format_history_for_model(self, turns: int = 3) -> str:
        """Format conversation history for model input"""
        recent_history = self.get_recent_history(turns)
        
        if not recent_history:
            return ""
            
        formatted_history = []
        for turn in recent_history:
            formatted_history.append(f"User: {turn.user_query}")
            formatted_history.append(f"Assistant: {turn.bot_response}")
            
        return "\n".join(formatted_history)
        
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
        
    def save_history(self, filepath: str):
        """Save conversation history to file"""
        try:
            history_data = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "turns": [asdict(turn) for turn in self.conversation_history]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Conversation history saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save conversation history: {e}")
            
    def load_history(self, filepath: str):
        """Load conversation history from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
                
            self.session_id = history_data.get("session_id")
            turns_data = history_data.get("turns", [])
            
            self.conversation_history.clear()
            for turn_data in turns_data:
                turn = ConversationTurn(**turn_data)
                self.conversation_history.append(turn)
                
            logger.info(f"Loaded {len(self.conversation_history)} conversation turns from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")


class ConversationManager:
    """Manages multiple conversation sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, ConversationMemory] = {}
        
    def get_or_create_session(self, session_id: str) -> ConversationMemory:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory()
            self.sessions[session_id].session_id = session_id
            logger.info(f"Created new conversation session: {session_id}")
            
        return self.sessions[session_id]
        
    def add_turn(self, session_id: str, turn: ConversationTurn):
        """Add turn to specific session"""
        session = self.get_or_create_session(session_id)
        session.add_turn(turn)
        
    def get_session_history(self, session_id: str, turns: int = 3) -> str:
        """Get formatted history for specific session"""
        session = self.get_or_create_session(session_id)
        return session.format_history_for_model(turns)
        
    def clear_session(self, session_id: str):
        """Clear specific session"""
        if session_id in self.sessions:
            self.sessions[session_id].clear_history()
            logger.info(f"Cleared session: {session_id}")
            
    def save_session(self, session_id: str, filepath: str):
        """Save specific session to file"""
        if session_id in self.sessions:
            self.sessions[session_id].save_history(filepath)
            
    def cleanup_old_sessions(self, max_sessions: int = 10):
        """Clean up old sessions to prevent memory issues"""
        if len(self.sessions) > max_sessions:
            # Remove oldest sessions (simple FIFO for now)
            sessions_to_remove = list(self.sessions.keys())[:-max_sessions]
            for session_id in sessions_to_remove:
                del self.sessions[session_id]
            logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")


# Global conversation manager instance
conversation_manager = ConversationManager()
