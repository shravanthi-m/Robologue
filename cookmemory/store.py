"""Single-document checkpoints plus idempotent decision projections.

One worker per session is supported. Replay uses compare_and_swap; save is
retained only for callers that deliberately replace an unmanaged checkpoint.
"""
import json
import os
import sqlite3
from pathlib import Path

from db import CheckpointConflict, MongoBackend


class SQLiteStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, state TEXT NOT NULL)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS decisions (
            session TEXT NOT NULL, event_id TEXT NOT NULL, decision TEXT NOT NULL,
            PRIMARY KEY (session, event_id))""")

    def load(self, session):
        row = self.db.execute("SELECT state FROM sessions WHERE id = ?", (session,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, session, state):
        with self.db:
            self.db.execute(
                "INSERT INTO sessions VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET state=excluded.state",
                (session, json.dumps(state)))

    def compare_and_swap(self, session, state, expected_revision):
        with self.db:
            if expected_revision is None:
                try:
                    self.db.execute("INSERT INTO sessions VALUES (?, ?)", (session, json.dumps(state)))
                except sqlite3.IntegrityError:
                    raise CheckpointConflict("Replay session already exists") from None
            else:
                result = self.db.execute("""UPDATE sessions SET state = ? WHERE id = ?
                    AND COALESCE(json_extract(state, '$.revision'), 0) = ?""",
                    (json.dumps(state), session, expected_revision))
                if result.rowcount != 1:
                    raise CheckpointConflict("Session changed; use one replay worker per session")

    def put_decision(self, session, decision):
        with self.db:
            self.db.execute("""INSERT INTO decisions VALUES (?, ?, ?)
                ON CONFLICT(session, event_id) DO UPDATE SET decision=excluded.decision""",
                (session, decision["event_id"], json.dumps(decision)))

    def decisions(self, session):
        return [json.loads(row[0]) for row in self.db.execute(
            "SELECT decision FROM decisions WHERE session = ? ORDER BY rowid", (session,))]

    def close(self):
        self.db.close()


class AtlasStore:
    def __init__(self):
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            raise ValueError("MONGODB_URI is required for the Atlas backend")
        self.backend = MongoBackend(uri, os.environ.get("MONGODB_DATABASE", "cookmemory"))
        self.client = self.backend.client
        self.collection = self.backend.db.sessions

    def load(self, session):
        doc = self.collection.find_one({"_id": session})
        return doc["state"] if doc else None

    def save(self, session, state):
        self.backend.save("sessions", {"_id": session, "revision": state.get("revision", 0), "state": state})

    def compare_and_swap(self, session, state, expected_revision):
        # Legacy checkpoint documents lacked a top-level revision. Atomic
        # migration only touches that field when it is still absent.
        if expected_revision == 0:
            self.collection.update_one(
                {"_id": session, "revision": {"$exists": False}}, {"$set": {"revision": 0}})
        self.backend.compare_and_swap("sessions", {
            "_id": session, "revision": state["revision"], "state": state}, expected_revision)

    def put_decision(self, session, decision):
        # Composite identity is encoded without ambiguous string delimiters.
        identity = json.dumps([session, decision["event_id"]], separators=(",", ":"))
        self.backend.save("recording_decisions", {
            "_id": identity, "session": session, "decision": decision})

    def decisions(self, session):
        docs = self.backend.db.recording_decisions.find({"session": session}).sort(
            [("decision.timestamp", 1), ("decision.event_id", 1)])
        return [d["decision"] for d in docs]

    def close(self):
        self.backend.close()
