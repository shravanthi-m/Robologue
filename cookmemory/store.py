"""Atomic single-document checkpoints; use one replay worker per session."""
import json
import os
import sqlite3
from pathlib import Path


class SQLiteStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, state TEXT NOT NULL)")

    def load(self, session):
        row = self.db.execute("SELECT state FROM sessions WHERE id = ?", (session,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, session, state):
        with self.db:
            self.db.execute("INSERT INTO sessions VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET state=excluded.state",
                            (session, json.dumps(state)))

    def close(self):
        self.db.close()


class AtlasStore:
    def __init__(self):
        from pymongo import MongoClient
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            raise ValueError("Export MONGODB_URI for your hackathon Atlas Sandbox.")
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.client.admin.command("ping")
        self.collection = self.client[os.environ.get("MONGODB_DATABASE", "cookmemory")].sessions

    def load(self, session):
        doc = self.collection.find_one({"_id": session})
        return doc["state"] if doc else None

    def save(self, session, state):
        self.collection.replace_one({"_id": session}, {"_id": session, "state": state}, upsert=True)

    def close(self):
        self.client.close()
