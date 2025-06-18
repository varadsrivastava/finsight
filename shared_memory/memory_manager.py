import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import chromadb
from chromadb.config import Settings
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Structure for memory entries"""
    id: str
    timestamp: datetime
    agent_name: str
    content_type: str  # "data", "analysis", "insight", "report_section"
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    tags: List[str]

class SharedMemoryManager:
    """Manages shared memory for multi-agent communication and data persistence"""
    
    def __init__(self, vector_db_path: str = "./data/vectordb", 
                 json_store_path: str = "./data/shared_memory.json"):
        self.vector_db_path = vector_db_path
        self.json_store_path = json_store_path
        self.lock = threading.Lock()
        
        # Initialize vector database
        os.makedirs(vector_db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="finsight_memory",
            metadata={"description": "FinSight multi-agent shared memory"}
        )
        
        # Initialize JSON store
        os.makedirs(os.path.dirname(json_store_path), exist_ok=True)
        self.json_store = self._load_json_store()
        
    def _load_json_store(self) -> Dict:
        """Load JSON store from disk"""
        try:
            if os.path.exists(self.json_store_path):
                with open(self.json_store_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load JSON store: {e}")
        return {"entries": [], "agents": {}, "sessions": {}}
    
    def _save_json_store(self):
        """Save JSON store to disk"""
        try:
            with open(self.json_store_path, 'w') as f:
                json.dump(self.json_store, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save JSON store: {e}")
    
    def store_entry(self, agent_name: str, content_type: str, 
                   content: Dict[str, Any], metadata: Dict[str, Any] = None,
                   tags: List[str] = None) -> str:
        """Store a new memory entry"""
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        if metadata is None:
            metadata = {}
        if tags is None:
            tags = []
            
        # Create memory entry
        entry = MemoryEntry(
            id=entry_id,
            timestamp=timestamp,
            agent_name=agent_name,
            content_type=content_type,
            content=content,
            metadata=metadata,
            tags=tags
        )
        
        with self.lock:
            # Store in JSON
            self.json_store["entries"].append(asdict(entry))
            
            # Update agent info
            if agent_name not in self.json_store["agents"]:
                self.json_store["agents"][agent_name] = {
                    "first_activity": timestamp.isoformat(),
                    "entry_count": 0
                }
            self.json_store["agents"][agent_name]["entry_count"] += 1
            self.json_store["agents"][agent_name]["last_activity"] = timestamp.isoformat()
            
            # Store in vector DB if content has text
            text_content = self._extract_text_content(content)
            if text_content:
                self.collection.add(
                    documents=[text_content],
                    metadatas=[{
                        "agent_name": agent_name,
                        "content_type": content_type,
                        "timestamp": timestamp.isoformat(),
                        **metadata
                    }],
                    ids=[entry_id]
                )
            
            self._save_json_store()
            
        logger.info(f"Stored memory entry {entry_id} from {agent_name}")
        return entry_id
    
    def _extract_text_content(self, content: Dict[str, Any]) -> str:
        """Extract searchable text from content"""
        text_parts = []
        
        for key, value in content.items():
            if isinstance(value, str):
                text_parts.append(f"{key}: {value}")
            elif isinstance(value, (int, float)):
                text_parts.append(f"{key}: {str(value)}")
            elif isinstance(value, list):
                text_parts.append(f"{key}: {', '.join(str(v) for v in value)}")
            elif isinstance(value, dict):
                nested_text = self._extract_text_content(value)
                if nested_text:
                    text_parts.append(f"{key}: {nested_text}")
                    
        return " | ".join(text_parts)
    
    def search_entries(self, query: str, agent_filter: str = None, 
                      content_type_filter: str = None, n_results: int = 10) -> List[Dict]:
        """Search entries using vector similarity"""
        try:
            # Build where clause for filtering
            where_clause = {}
            if agent_filter:
                where_clause["agent_name"] = agent_filter
            if content_type_filter:
                where_clause["content_type"] = content_type_filter
            
            # Perform vector search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            # Return formatted results
            entries = []
            if results["ids"] and results["ids"][0]:
                for i, entry_id in enumerate(results["ids"][0]):
                    entries.append({
                        "id": entry_id,
                        "distance": results["distances"][0][i] if results["distances"] else None,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "content": results["documents"][0][i] if results["documents"] else ""
                    })
            
            return entries
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_entry_by_id(self, entry_id: str) -> Optional[Dict]:
        """Get a specific entry by ID"""
        with self.lock:
            for entry in self.json_store["entries"]:
                if entry["id"] == entry_id:
                    return entry
        return None
    
    def get_agent_entries(self, agent_name: str, limit: int = 50) -> List[Dict]:
        """Get all entries from a specific agent"""
        with self.lock:
            agent_entries = [
                entry for entry in self.json_store["entries"] 
                if entry["agent_name"] == agent_name
            ]
            # Sort by timestamp descending
            agent_entries.sort(key=lambda x: x["timestamp"], reverse=True)
            return agent_entries[:limit]
    
    def get_recent_entries(self, hours: int = 24, limit: int = 100) -> List[Dict]:
        """Get recent entries within specified hours"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        with self.lock:
            recent_entries = []
            for entry in self.json_store["entries"]:
                entry_time = datetime.fromisoformat(entry["timestamp"]).timestamp()
                if entry_time > cutoff_time:
                    recent_entries.append(entry)
            
            # Sort by timestamp descending
            recent_entries.sort(key=lambda x: x["timestamp"], reverse=True)
            return recent_entries[:limit]
    
    def update_entry(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing entry"""
        with self.lock:
            for i, entry in enumerate(self.json_store["entries"]):
                if entry["id"] == entry_id:
                    # Update the entry
                    entry.update(updates)
                    entry["metadata"]["last_updated"] = datetime.now().isoformat()
                    
                    self._save_json_store()
                    
                    # Update vector DB if needed
                    if "content" in updates:
                        text_content = self._extract_text_content(updates["content"])
                        if text_content:
                            self.collection.update(
                                ids=[entry_id],
                                documents=[text_content]
                            )
                    
                    return True
            return False
    
    def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry"""
        with self.lock:
            # Remove from JSON store
            original_count = len(self.json_store["entries"])
            self.json_store["entries"] = [
                entry for entry in self.json_store["entries"] 
                if entry["id"] != entry_id
            ]
            
            if len(self.json_store["entries"]) < original_count:
                self._save_json_store()
                
                # Remove from vector DB
                try:
                    self.collection.delete(ids=[entry_id])
                except Exception as e:
                    logger.warning(f"Could not delete from vector DB: {e}")
                
                return True
            
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        with self.lock:
            total_entries = len(self.json_store["entries"])
            agent_stats = {}
            
            for agent_name, info in self.json_store["agents"].items():
                agent_stats[agent_name] = {
                    "entry_count": info["entry_count"],
                    "first_activity": info["first_activity"],
                    "last_activity": info.get("last_activity", "Unknown")
                }
            
            # Content type distribution
            content_types = {}
            for entry in self.json_store["entries"]:
                content_type = entry["content_type"]
                content_types[content_type] = content_types.get(content_type, 0) + 1
            
            return {
                "total_entries": total_entries,
                "agent_stats": agent_stats,
                "content_type_distribution": content_types,
                "vector_db_size": self.collection.count(),
            }
    
    def clear_memory(self, confirm: bool = False):
        """Clear all memory (use with caution)"""
        if not confirm:
            raise ValueError("Must set confirm=True to clear memory")
            
        with self.lock:
            # Clear JSON store
            self.json_store = {"entries": [], "agents": {}, "sessions": {}}
            self._save_json_store()
            
            # Clear vector DB
            self.collection.delete()
            
            logger.warning("All memory cleared")