from __future__ import annotations

import atexit
import gc
import logging
import sqlite3
import threading

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Hashable

log = logging.getLogger(__name__)

@dataclass
class ResourceSlot:
    signature: Hashable | None = None
    value: Any = None
    lock : threading.RLock = field(default_factory=threading.RLock)

def close_resource(resource: Any) -> None:
    if resource is None:
        return
    for method_name in ("close", "shutdown", "unload"):
        method = getattr(resource, method_name, None)
        if not callable(method):
            continue
        try:
            method()
        except Exception:
            log.exception("[RESOURCE] failed close %s using %s", type(resource).__name__, method_name)
        break

def path_signature(path: str | Path, marker_files: tuple[str, ...] = ()) -> tuple:
    root = Path(path).expanduser().resolve()
    stamps: list[tuple] = []

    if root.is_file():
        targets = [root]
    else:
        targets = [root / marker for marker in marker_files]

    for target in targets:
        try:
            stat = target.stat()
            stamps.append((str(target), stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            stamps.append((str(target), None, None))

    return (str(root), tuple(stamps))

class ResourceManager:
    def __init__(self) -> None:
        self._slots: dict[Hashable, ResourceSlot] = {}
        self._slots_lock = threading.RLock()
        self._thread_local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._connections_lock = (threading.RLock())

    def _get_slot(self, name: Hashable) -> ResourceSlot:
        with self._slots_lock:
            slot = self._slots.get(name)
            if slot is None:
                slot = ResourceSlot()
                self._slots[name] = slot
            return slot

    def get(self, name: Hashable, signature: Hashable, factory: Callable[[], Any]) -> Any:
        slot = self._get_slot(name)

        if (slot.value is not None and slot.signature == signature):
            log.debug("[RESOURCE] reuse name=%r", name)
            return slot.value

        with slot.lock:
            if (slot.value is not None and slot.signature == signature):
                return slot.value

            if slot.value is not None:
                log.info("[RESOURCE] signature changed; reloading name=%r", name)
                old_resource = slot.value
                slot.value = None
                slot.signature = None
                close_resource(old_resource)
                del old_resource
                gc.collect()
            else:
                log.info("[RESOURCE] loading name=%r", name)

            resource = factory()
            slot.value = resource
            slot.signature = signature
            return resource

    def get_sqlite_connection(self, db_path: str | Path) -> sqlite3.Connection:
        path = Path(db_path).expanduser().resolve()

        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            signature = (None, None)

        connections = getattr(self._thread_local, "sqlite_connections", None)

        if connections is None:
            connections = {}
            self._thread_local.sqlite_connections = connections

        key = str(path)
        existing = connections.get(key)

        if existing is not None:
            old_signature, connection = existing

            if old_signature == signature:
                try:
                    connection.execute("SELECT 1")
                    return connection
                except sqlite3.Error:
                    pass

            try:
                connection.close()
            except sqlite3.Error:
                pass

            connections.pop(key, None)

        connection = sqlite3.connect(f"file:{path}?mode=ro&cache=shared", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connections[key] = (signature, connection)

        with self._connections_lock:
            self._connections.append(connection)

        log.info("[RESOURCE] opened SQLite thread=%s path=%s", threading.get_ident(), path)

        return connection

    def get_bm25(self, db_path: str | Path, factory: Callable[[sqlite3.Connection], Any]) -> Any:
        connection = self.get_sqlite_connection(db_path)
        bm25_instances = getattr(self._thread_local, "bm25_instances", None)

        if bm25_instances is None:
            bm25_instances = {}
            self._thread_local.bm25_instances = bm25_instances

        key = str(Path(db_path).expanduser().resolve())
        existing = bm25_instances.get(key)

        if existing is not None and existing[0] is connection:
            return existing[1]

        retriever = factory(connection)
        bm25_instances[key] = (connection, retriever)

        log.info("[RESOURCE] created BM25 thread=%s path=%s", threading.get_ident(), key)

        return retriever

    def invalidate(self, name: Hashable | None = None) -> None:
        with self._slots_lock:
            if name is None:
                items = list(self._slots.items())
            else:
                slot = self._slots.get(name)
                items = [(name, slot)] if slot is not None else []

        for resource_name, slot in items:
            with slot.lock:
                resource = slot.value
                slot.value = None
                slot.signature = None
                close_resource(resource)
                log.info("[RESOURCE] invalidated name=%r", resource_name)

        gc.collect()

    def status(self) -> dict[str, Any]:
        with self._slots_lock:
            resources = {str(name): slot.value is not None for name, slot in self._slots.items()}

        sqlite_connections = getattr(self._thread_local, "sqlite_connections", {})
        bm25_instances = getattr(self._thread_local, "bm25_instances", {})

        return {
            "resources": resources,
            "sqlite_connections": len(sqlite_connections),
            "bm25_instances": len(bm25_instances),
        }

    def close_all(self) -> None:
        self.invalidate()

        with self._connections_lock:
            connections = list(self._connections)
            self._connections.clear()

        for connection in connections:
            try:
                connection.close()
            except sqlite3.Error:
                pass

        log.info("[RESOURCE] all resources closed")


RESOURCES = ResourceManager()
atexit.register(RESOURCES.close_all)