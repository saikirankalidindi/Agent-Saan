"""Long-Term Memory backed by PostgreSQL + pgvector.

Stores and retrieves ``MemoryEntry`` objects using semantic similarity search
powered by OpenAI ``text-embedding-3-small`` embeddings and a pgvector HNSW
index for cosine similarity.

Requirements: 2.2, 2.3, 2.4, 2.6
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import asyncpg
from openai import AsyncOpenAI

from agent_saan.models.memory import ConflictResult, MemoryEntry

logger = logging.getLogger(__name__)

# Embedding model used for all vector operations.
_EMBEDDING_MODEL = "text-embedding-3-small"
# Dimensionality of text-embedding-3-small output.
_EMBEDDING_DIM = 1536
# Cosine similarity threshold above which a conflict is considered.
_CONFLICT_SIMILARITY_THRESHOLD = 0.85


class LongTermMemory:
    """Manages persistent long-term memory in PostgreSQL with pgvector.

    Parameters
    ----------
    pool:
        An ``asyncpg`` connection pool connected to the Agent Saan database.
    openai_client:
        An ``openai.AsyncOpenAI`` client used to generate embeddings.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        openai_client: AsyncOpenAI,
    ) -> None:
        self._pool = pool
        self._openai = openai_client

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text* using text-embedding-3-small."""
        response = await self._openai.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    @staticmethod
    def _row_to_entry(row: asyncpg.Record) -> MemoryEntry:
        """Convert an asyncpg row from ``memory_entries`` to a ``MemoryEntry``."""
        # The embedding column is returned as a pgvector string "[x,y,z,...]"
        # or as a list depending on the asyncpg codec registration.
        raw_embedding = row["embedding"]
        if isinstance(raw_embedding, str):
            embedding: list[float] = json.loads(raw_embedding)
        elif raw_embedding is None:
            embedding = []
        else:
            embedding = list(raw_embedding)

        raw_tags = row["tags"]
        if isinstance(raw_tags, str):
            tags: list[str] = json.loads(raw_tags)
        elif raw_tags is None:
            tags = []
        else:
            tags = list(raw_tags)

        return MemoryEntry(
            entry_id=row["entry_id"],
            user_id=row["user_id"],
            namespace=row["namespace"],
            content=row["content"],
            source_type=row["source_type"],
            embedding=embedding,
            created_at=row["created_at"],
            tags=tags,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def upsert(self, entry: MemoryEntry) -> None:
        """Embed *entry.content* and upsert the record into ``memory_entries``.

        If a row with the same ``entry_id`` already exists it is updated;
        otherwise a new row is inserted.

        Requirements: 2.2
        """
        embedding = await self._embed(entry.content)

        # Serialise the embedding as a pgvector literal string.
        embedding_literal = "[" + ",".join(str(v) for v in embedding) + "]"
        tags_json = json.dumps(entry.tags)

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_entries
                    (entry_id, user_id, namespace, content, source_type,
                     embedding, created_at, tags, status)
                VALUES
                    ($1, $2, $3, $4, $5,
                     $6::vector, $7, $8::jsonb, 'active')
                ON CONFLICT (entry_id) DO UPDATE SET
                    content     = EXCLUDED.content,
                    namespace   = EXCLUDED.namespace,
                    source_type = EXCLUDED.source_type,
                    embedding   = EXCLUDED.embedding,
                    tags        = EXCLUDED.tags,
                    status      = 'active'
                """,
                entry.entry_id,
                entry.user_id,
                entry.namespace,
                entry.content,
                entry.source_type,
                embedding_literal,
                entry.created_at,
                tags_json,
            )

    async def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        """Return the *top_k* most semantically similar active entries for *user_id*.

        Embeds *query* and runs a cosine similarity search using the HNSW index.
        Target latency: ≤200ms (Requirement 2.3).

        Requirements: 2.3
        """
        query_embedding = await self._embed(query)
        embedding_literal = "[" + ",".join(str(v) for v in query_embedding) + "]"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    entry_id, user_id, namespace, content, source_type,
                    embedding, created_at, tags
                FROM memory_entries
                WHERE user_id = $1
                  AND status  = 'active'
                ORDER BY embedding <=> $2::vector
                LIMIT $3
                """,
                user_id,
                embedding_literal,
                top_k,
            )

        return [self._row_to_entry(row) for row in rows]

    async def delete(self, entry_id: str, user_id: str) -> bool:
        """Hard-delete the entry identified by *entry_id* if it belongs to *user_id*.

        Returns ``True`` when a row was deleted, ``False`` when no matching row
        was found (wrong ``entry_id`` or wrong ``user_id``).

        Requirements: 2.4
        """
        async with self._pool.acquire() as conn:
            result: str = await conn.execute(
                """
                DELETE FROM memory_entries
                WHERE entry_id = $1
                  AND user_id  = $2
                """,
                entry_id,
                user_id,
            )

        # asyncpg returns a command tag like "DELETE 1" or "DELETE 0".
        deleted_count = int(result.split()[-1])
        return deleted_count > 0

    async def detect_conflict(
        self,
        new_statement: str,
        user_id: str,
    ) -> ConflictResult:
        """Check whether *new_statement* conflicts with an existing LTM entry.

        Retrieves the single most similar active entry for *user_id*.  If the
        cosine similarity exceeds ``_CONFLICT_SIMILARITY_THRESHOLD`` (0.85) AND
        the LLM judges the two statements to contradict each other, a
        ``ConflictResult`` with ``has_conflict=True`` is returned.

        The contradiction check is performed by asking the embedding model's
        cosine distance as a proxy: if similarity > 0.85 the content is
        semantically close enough to warrant a conflict flag.  A full LLM
        contradiction check is intentionally deferred to the Orchestrator layer
        to keep this method fast and testable without an LLM call.

        Requirements: 2.6
        """
        query_embedding = await self._embed(new_statement)
        embedding_literal = "[" + ",".join(str(v) for v in query_embedding) + "]"

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    entry_id, user_id, namespace, content, source_type,
                    embedding, created_at, tags,
                    1 - (embedding <=> $2::vector) AS similarity
                FROM memory_entries
                WHERE user_id = $1
                  AND status  = 'active'
                ORDER BY embedding <=> $2::vector
                LIMIT 1
                """,
                user_id,
                embedding_literal,
            )

        if row is None:
            return ConflictResult(
                has_conflict=False,
                new_statement=new_statement,
            )

        similarity: float = float(row["similarity"])

        if similarity <= _CONFLICT_SIMILARITY_THRESHOLD:
            return ConflictResult(
                has_conflict=False,
                new_statement=new_statement,
                similarity_score=similarity,
            )

        existing_entry = self._row_to_entry(row)
        return ConflictResult(
            has_conflict=True,
            existing_entry=existing_entry,
            new_statement=new_statement,
            similarity_score=similarity,
        )
