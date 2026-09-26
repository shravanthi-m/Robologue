"""Explicit backends and run-scoped storage without changing frozen contracts."""
from __future__ import annotations
import json
import os
import re
import sqlite3
import uuid
from pathlib import Path

class BackendUnavailable(RuntimeError):
    """A requested backend is unavailable; errors never include a URI."""

class DocumentExists(RuntimeError):
    """An immutable document already exists."""

class CheckpointConflict(RuntimeError):
    """Another writer advanced this run."""

def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((jsonable(v) for v in obj), key=repr)
    return obj

def document(doc):
    doc = dict(jsonable(doc))
    doc["_id"] = str(doc.get("_id") or doc.get("trace_id") or doc.get("mutation_id")
                     or doc.get("version_id") or doc.get("lesson_id") or uuid.uuid4().hex)
    return doc

def load_env_file(path):
    """Load KEY=VALUE without expansion or overwriting existing environment."""
    values = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError("Invalid environment file; expected KEY=VALUE lines")
        value = value.strip()
        if value[:1] in ('"', "'"):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError("Invalid quoted value in environment file")
            value = value[1:-1]
        values[key] = value
    for key, value in values.items():
        os.environ.setdefault(key, value)
    # Support the name used in the team's Sandbox credentials file.
    if not os.environ.get("MONGODB_URI") and os.environ.get("MongoDB_Connection_String"):
        os.environ["MONGODB_URI"] = os.environ["MongoDB_Connection_String"]

class SQLiteBackend:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.backend_name = f"sqlite:{path}"

    def _table(self, coll):
        if not coll.replace("_", "").isalnum():
            raise ValueError(f"bad collection name: {coll}")
        self.conn.execute(f'CREATE TABLE IF NOT EXISTS "{coll}" (id TEXT PRIMARY KEY, doc TEXT)')
        return coll

    def save(self, coll, doc):
        table, doc = self._table(coll), document(doc)
        with self.conn:
            self.conn.execute(f'INSERT OR REPLACE INTO "{table}" (id, doc) VALUES (?, ?)',
                              (doc["_id"], json.dumps(doc)))
        return doc["_id"]

    def insert(self, coll, doc):
        table, doc = self._table(coll), document(doc)
        try:
            with self.conn:
                self.conn.execute(f'INSERT INTO "{table}" (id, doc) VALUES (?, ?)',
                                  (doc["_id"], json.dumps(doc)))
        except sqlite3.IntegrityError:
            raise DocumentExists(f"Document already exists in {coll}") from None
        return doc["_id"]

    @staticmethod
    def _matches(doc, query):
        for key, expected in (query or {}).items():
            actual = doc.get(key)
            if isinstance(expected, dict) and set(expected) == {"$regex"}:
                if not isinstance(actual, str) or not re.search(expected["$regex"], actual):
                    return False
            elif actual != expected:
                return False
        return True

    def find(self, coll, query=None, limit=500):
        if limit is not None and limit < 1:
            return []
        table, docs = self._table(coll), []
        # Filter before limiting; later matching documents must remain visible.
        for row in self.conn.execute(f'SELECT doc FROM "{table}" ORDER BY rowid'):
            doc = json.loads(row["doc"])
            if self._matches(doc, query):
                docs.append(doc)
                if limit is not None and len(docs) >= limit:
                    break
        return docs

    def find_one(self, coll, query):
        docs = self.find(coll, query, limit=1)
        return docs[0] if docs else None

    def delete_where(self, coll, query):
        ids = [d["_id"] for d in self.find(coll, query, limit=None)]
        with self.conn:
            self.conn.executemany(f'DELETE FROM "{coll}" WHERE id = ?', ((i,) for i in ids))

    def clear(self, coll):
        self.delete_where(coll, {})

    def compare_and_swap(self, coll, doc, expected_revision):
        if expected_revision is None:
            try:
                return self.insert(coll, doc)
            except DocumentExists:
                raise CheckpointConflict("Run already exists; use --resume") from None
        table, doc = self._table(coll), document(doc)
        with self.conn:
            result = self.conn.execute(
                f"""UPDATE "{table}" SET doc = ? WHERE id = ?
                    AND json_extract(doc, '$.revision') = ?""",
                (json.dumps(doc), doc["_id"], expected_revision))
        if result.rowcount != 1:
            raise CheckpointConflict("Checkpoint changed; only one worker per run is supported")
        return doc["_id"]

    def close(self):
        self.conn.close()

class MongoBackend:
    def __init__(self, uri, db_name):
        try:
            from pymongo import MongoClient
        except ImportError:
            raise BackendUnavailable("Install the Atlas extra: pip install -e '.[atlas]'") from None
        self.client = None
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=8000,
                                      connectTimeoutMS=8000, socketTimeoutMS=8000,
                                      w="majority", retryWrites=True)
            self.db = self.client[db_name]
            self.backend_name = f"atlas:{db_name}"
            self.client.admin.command("ping")
        except Exception:
            if self.client is not None:
                self.client.close()
            raise BackendUnavailable(
                "Atlas connection failed; check Sandbox credentials, network access and allowlist"
            ) from None

    def save(self, coll, doc):
        doc = document(doc)
        self.db[coll].replace_one({"_id": doc["_id"]}, doc, upsert=True)
        return doc["_id"]

    def insert(self, coll, doc):
        from pymongo.errors import DuplicateKeyError
        doc = document(doc)
        try:
            self.db[coll].insert_one(doc)
        except DuplicateKeyError:
            raise DocumentExists(f"Document already exists in {coll}") from None
        return doc["_id"]

    def find(self, coll, query=None, limit=500):
        if limit is not None and limit < 1:
            return []
        cursor = self.db[coll].find(query or {})
        if limit is not None:
            cursor = cursor.limit(limit)
        return list(cursor)

    def find_one(self, coll, query):
        return self.db[coll].find_one(query)

    def delete_where(self, coll, query):
        self.db[coll].delete_many(query)

    def clear(self, coll):
        self.delete_where(coll, {})

    def compare_and_swap(self, coll, doc, expected_revision):
        if expected_revision is None:
            try:
                return self.insert(coll, doc)
            except DocumentExists:
                raise CheckpointConflict("Run already exists; use --resume") from None
        doc = document(doc)
        result = self.db[coll].replace_one(
            {"_id": doc["_id"], "revision": expected_revision}, doc, upsert=False)
        if result.matched_count != 1:
            raise CheckpointConflict("Checkpoint changed; only one worker per run is supported")
        return doc["_id"]

    def close(self):
        self.client.close()

class RunBackend:
    """Scope every collection by storage ID; application field names are stable."""
    def __init__(self, backend, run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
            raise ValueError("run_id must be 1-64 letters/digits/underscores/hyphens")
        self.backend, self.run_id = backend, run_id
        self.prefix = f"run:{run_id}:"
        self.backend_name = f"{backend.backend_name} run={run_id}"

    def _write(self, doc):
        doc = document(doc)
        doc["_id"] = self.prefix + doc["_id"]
        return doc

    def _read(self, doc):
        if doc is not None:
            doc = dict(doc)
            doc["_id"] = doc["_id"][len(self.prefix):]
        return doc

    def _query(self, query=None):
        query = dict(query or {})
        if "_id" in query:
            if not isinstance(query["_id"], str):
                raise ValueError("Scoped _id queries must use an exact string")
            query["_id"] = self.prefix + query["_id"]
        else:
            query["_id"] = {"$regex": "^" + re.escape(self.prefix)}
        return query

    def save(self, coll, doc):
        return self.backend.save(coll, self._write(doc))[len(self.prefix):]

    def insert(self, coll, doc):
        return self.backend.insert(coll, self._write(doc))[len(self.prefix):]

    def find(self, coll, query=None, limit=500):
        return [self._read(d) for d in self.backend.find(coll, self._query(query), limit)]

    def find_one(self, coll, query):
        return self._read(self.backend.find_one(coll, self._query(query)))

    def clear(self, coll):
        self.backend.delete_where(coll, self._query())

    def compare_and_swap(self, coll, doc, expected_revision):
        return self.backend.compare_and_swap(coll, self._write(doc), expected_revision)[len(self.prefix):]

    def close(self):
        self.backend.close()

def get_db(mode=None, *, env_file=None, run_id=None):
    if env_file:
        load_env_file(env_file)
    mode = mode or os.environ.get("HARNESS_BACKEND", "auto")
    if mode not in ("auto", "atlas", "local"):
        raise ValueError("backend must be auto, atlas or local")
    uri = os.environ.get("MONGODB_URI")
    if mode == "atlas" and not uri:
        raise BackendUnavailable("MONGODB_URI is required for the Atlas backend")
    if mode == "atlas" or (mode == "auto" and uri):
        try:
            backend = MongoBackend(uri, os.environ.get("MONGODB_DATABASE", "cookmemory"))
        except BackendUnavailable:
            raise
        except Exception:
            raise BackendUnavailable("Atlas connection failed; check Sandbox configuration") from None
    else:
        backend = SQLiteBackend(os.environ.get("DEV_DB", "work/dev.sqlite"))
    return RunBackend(backend, run_id) if run_id else backend
