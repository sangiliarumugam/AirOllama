import os
import sqlite3
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.path.expanduser("~/.airollama/airollama.db")

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        path TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        project_id INTEGER,
        title TEXT NOT NULL,
        model TEXT,
        role TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        thought TEXT,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
    )
    """)
    
    cursor.execute("SELECT COUNT(*) FROM projects")
    if cursor.fetchone()[0] == 0:
        default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        now = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO projects (name, path, created_at, updated_at) VALUES (?, ?, ?, ?)",
            ("AirOllama Core", default_path, now, now)
        )
    
    conn.commit()
    conn.close()

# --- Project CRUD ---

def list_projects() -> List[Dict[str, Any]]:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM projects ORDER BY updated_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def create_project(name: str, path: str) -> Dict[str, Any]:
    init_db()
    abs_path = os.path.abspath(os.path.expanduser(path))
    now = datetime.now().isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO projects (name, path, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (name, abs_path, now, now)
    )
    project_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"id": project_id, "name": name, "path": abs_path, "created_at": now, "updated_at": now}

def delete_project(project_id: int) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()
    return True

def get_project(project_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

# --- Conversation CRUD ---

def list_conversations(project_id: Optional[int] = None) -> List[Dict[str, Any]]:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    if project_id is not None:
        cursor.execute("SELECT * FROM conversations WHERE project_id = ? ORDER BY updated_at DESC", (project_id,))
    else:
        cursor.execute("SELECT * FROM conversations ORDER BY updated_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def create_conversation(project_id: Optional[int], title: str, model: str = "", role: str = "") -> Dict[str, Any]:
    init_db()
    conv_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    if not title:
        title = "New Coding Thread"
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (id, project_id, title, model, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (conv_id, project_id, title, model, role, now, now)
    )
    conn.commit()
    conn.close()
    return {"id": conv_id, "project_id": project_id, "title": title, "model": model, "role": role, "created_at": now, "updated_at": now}

def delete_conversation(conversation_id: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()
    return True

def update_conversation(conversation_id: str, title: Optional[str] = None, model: Optional[str] = None, role: Optional[str] = None) -> Optional[Dict[str, Any]]:
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None
    conv = dict(row)
    new_title = title if title is not None else conv["title"]
    new_model = model if model is not None else conv["model"]
    new_role = role if role is not None else conv["role"]
    now = datetime.now().isoformat()

    cursor.execute(
        "UPDATE conversations SET title = ?, model = ?, role = ?, updated_at = ? WHERE id = ?",
        (new_title, new_model, new_role, now, conversation_id)
    )
    conn.commit()
    conn.close()
    conv["title"] = new_title
    conv["model"] = new_model
    conv["role"] = new_role
    conv["updated_at"] = now
    return conv

# --- Message CRUD ---

def get_conversation_messages(conversation_id: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC", (conversation_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def add_message(conversation_id: str, role: str, content: str, thought: Optional[str] = None) -> Dict[str, Any]:
    init_db()
    now = datetime.now().isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (conversation_id, role, content, thought, timestamp) VALUES (?, ?, ?, ?, ?)",
        (conversation_id, role, content, thought, now)
    )
    msg_id = cursor.lastrowid
    cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    conn.commit()
    conn.close()
    return {"id": msg_id, "conversation_id": conversation_id, "role": role, "content": content, "thought": thought, "timestamp": now}
