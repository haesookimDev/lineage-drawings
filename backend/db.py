import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path


def data_dir():
    path = Path(os.environ.get('DATA_DIR', './data')).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def connect():
    db = sqlite3.connect(data_dir() / 'lineage.sqlite3', timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def initialize():
    with connect() as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript('''
        CREATE TABLE IF NOT EXISTS documents (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, format TEXT NOT NULL,
          status TEXT NOT NULL, error TEXT, warnings TEXT NOT NULL DEFAULT '[]',
          created TEXT NOT NULL, updated TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS pages (
          id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
          number INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
          coordinate_system TEXT NOT NULL, transform TEXT NOT NULL DEFAULT '{}');
        CREATE TABLE IF NOT EXISTS concepts (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, definition TEXT NOT NULL,
          scope TEXT NOT NULL, unit TEXT NOT NULL DEFAULT '',
          created TEXT NOT NULL, updated TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1,
          UNIQUE(name, scope));
        CREATE TABLE IF NOT EXISTS elements (
          id TEXT PRIMARY KEY, page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
          label TEXT NOT NULL, kind TEXT NOT NULL, bbox TEXT NOT NULL,
          source TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'needs_review',
          properties TEXT NOT NULL DEFAULT '{}', concept_id TEXT REFERENCES concepts(id));
        CREATE TABLE IF NOT EXISTS connections (
          id TEXT PRIMARY KEY, page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
          from_id TEXT NOT NULL REFERENCES elements(id) ON DELETE CASCADE,
          to_id TEXT NOT NULL REFERENCES elements(id) ON DELETE CASCADE,
          kind TEXT NOT NULL, source TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'needs_review',
          CHECK(from_id != to_id));
        CREATE TABLE IF NOT EXISTS audits (
          id INTEGER PRIMARY KEY AUTOINCREMENT, target_id TEXT NOT NULL, action TEXT NOT NULL,
          before_value TEXT, after_value TEXT, created TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sessions (
          token_hash TEXT PRIMARY KEY, csrf TEXT NOT NULL, expires REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_page_doc ON pages(document_id);
        CREATE INDEX IF NOT EXISTS idx_element_page ON elements(page_id);
        CREATE INDEX IF NOT EXISTS idx_connection_page ON connections(page_id);
        ''')
        db.execute("UPDATE documents SET status='queued' WHERE status='processing'")


def decode(row):
    result = dict(row)
    for field in ('bbox', 'properties', 'warnings', 'transform', 'before_value', 'after_value'):
        if field in result and result[field] is not None:
            result[field] = json.loads(result[field])
    return result
