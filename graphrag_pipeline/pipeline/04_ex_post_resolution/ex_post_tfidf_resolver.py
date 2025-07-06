"""
Resolve duplicate entities in Neo4j using TF‑IDF + cosine similarity.
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Tuple

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import neo4j

# -----------------------------------------------------------------------------
# 0.  Path & imports -----------------------------------------------------------
# -----------------------------------------------------------------------------
script_dir = Path(__file__).parent
graphrag_pipeline_dir = script_dir.parent.parent
if graphrag_pipeline_dir not in sys.path:
    sys.path.append(str(graphrag_pipeline_dir))


# -----------------------------------------------------------------------------
# 1.  TF‑IDF resolver class ----------------------------------------------------
# -----------------------------------------------------------------------------
class TFIDFMatchResolver:
    """
    Merge nodes whose concatenated text properties are similar according
    to TF‑IDF cosine similarity.

    Parameters
    ----------
    driver : neo4j.AsyncDriver
        Neo4j async driver (bolt or neo4j scheme).
    filter_query : str | None
        Cypher query that MUST return a collection of nodes as `n`.
        Example: ``MATCH (n:Person) WHERE n.document_id = $docId RETURN n``.
        If None, uses MATCH (n) RETURN n.
    resolve_properties : List[str]
        Textual properties to concatenate for vectorisation.
    similarity_threshold : float
        Cosine similarity ∈ (0, 1] above which two nodes are merged.
    neo4j_database : str
        Neo4j DB name (default "neo4j").
    """
    def __init__(
        self,
        driver: neo4j.AsyncDriver,
        filter_query: str | None,
        resolve_properties: List[str],
        similarity_threshold: float = 0.9,
        neo4j_database: str = "neo4j",
    ) -> None:
        self.driver = driver
        self.filter_query = filter_query
        self.resolve_properties = resolve_properties
        self.similarity_threshold = similarity_threshold
        self.db = neo4j_database

    # -------------------- public API --------------------
    async def run(self) -> Dict[str, int]:
        """
        Execute end‑to‑end resolution.

        Returns
        -------
        dict with counts:
            { 'nodes_examined': int,
              'pairs_above_threshold': int,
              'merged_pairs': int }
        """
        # 1. Pull candidate nodes
        nodes = await self._fetch_nodes()
        if not nodes:
            return {
                "nodes_examined": 0,
                "pairs_above_threshold": 0,
                "merged_pairs": 0,
            }

        ids, docs = zip(*nodes)                            # unzip
        # 2. Vectorise
        tfidf = TfidfVectorizer(stop_words="english").fit_transform(docs)
        sims = cosine_similarity(tfidf)

        # 3. Build unique candidate pairs (i<j) exceeding threshold
        pairs_to_merge: List[Tuple[int, int]] = []
        n = len(ids)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if sims[i, j] >= self.similarity_threshold:
                    pairs_to_merge.append((ids[i], ids[j]))

        # 4. Merge pairs in Neo4j (with APOC)
        merged_count = await self._merge_pairs(pairs_to_merge)

        return {
            "nodes_examined": n,
            "pairs_above_threshold": len(pairs_to_merge),
            "merged_pairs": merged_count,
        }

    # -------------------- helpers --------------------
    async def _fetch_nodes(self) -> List[Tuple[int, str]]:
        """
        Returns a list of (node_id, concatenated_text) tuples.
        """
        # Use predefined literal strings to avoid type issues
        DEFAULT_QUERY = "MATCH (n) RETURN n"
        
        async with self.driver.session(database=self.db) as session:
            records = []
            try:
                if self.filter_query:
                    # For custom queries, we'll need to use a workaround
                    # This is a limitation of the current Neo4j typing
                    query = self.filter_query.strip()
                    # Cast to any to bypass typing restrictions
                    result_cursor = await session.run(query)  # type: ignore
                else:
                    result_cursor = await session.run(DEFAULT_QUERY)
                
                # Try to get data using different methods
                try:
                    records = await result_cursor.data()
                except AttributeError:
                    # Fallback for older versions
                    async for record in result_cursor:
                        records.append(record.data())
                        
            except Exception as e:
                print(f"Warning: Query execution failed: {e}")
                return []
                    
            result = []
            for rec in records:
                try:
                    node = rec["n"]
                    text_parts = [
                        str(node.get(p, "")).strip() for p in self.resolve_properties
                    ]
                    combined = " ".join(text_parts).lower()
                    if combined:
                        result.append((node.id, combined))
                except Exception as e:
                    print(f"Warning: Failed to process node: {e}")
                    continue
            return result

    async def _merge_pairs(self, pairs: List[Tuple[int, int]]) -> int:
        """
        Merge each pair with APOC.  Returns number of merges executed.
        """
        if not pairs:
            return 0

        MERGE_QUERY = """
        MATCH (a) WHERE id(a) = $id1
        MATCH (b) WHERE id(b) = $id2
        WITH [a,b] AS nodes
        CALL apoc.refactor.mergeNodes(nodes, {properties:"discard"}) YIELD node
        RETURN id(node) AS kept
        """
        
        async with self.driver.session(database=self.db) as session:
            successful_merges = 0
            for id1, id2 in pairs:
                try:
                    # Use type: ignore to bypass typing restrictions
                    await session.run(MERGE_QUERY, id1=id1, id2=id2)  # type: ignore
                    successful_merges += 1
                except Exception as e:
                    print(f"Warning: Failed to merge nodes {id1} and {id2}: {e}")
                    continue
        return successful_merges


# -----------------------------------------------------------------------------
# 2.  Main entry point ---------------------------------------------------------
# -----------------------------------------------------------------------------
async def main() -> Dict[str, int]:
    """Run TF‑IDF entity resolution based on your config files."""
    # ---------- load config -----------
    config_files_path = graphrag_pipeline_dir / "config_files"
    load_dotenv(config_files_path / ".env", override=True)

    # Load only the required config file
    with open(config_files_path / "kg_building_config.json") as f:
        build_config = json.load(f)

    # ---------- Neo4j credentials ----------
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    # Validate required environment variables
    if not neo4j_uri:
        raise ValueError("NEO4J_URI environment variable is required")
    if not neo4j_username:
        raise ValueError("NEO4J_USERNAME environment variable is required")
    if not neo4j_password:
        raise ValueError("NEO4J_PASSWORD environment variable is required")

    tfidf_cfg = build_config["entity_resolution_config"]["TFIDFMatchResolver_config"]

    # ---------- run resolver ----------
    async with neo4j.AsyncGraphDatabase.driver(
        neo4j_uri, auth=(neo4j_username, neo4j_password)
    ) as driver:

        resolver = TFIDFMatchResolver(
            driver,
            filter_query=tfidf_cfg["filter_query"],
            resolve_properties=tfidf_cfg["resolve_properties"],
            similarity_threshold=tfidf_cfg.get("similarity_threshold", 0.75),
            neo4j_database=tfidf_cfg.get("neo4j_database", "neo4j"),
        )

        print(f"Running TFIDFMatchResolver (threshold={resolver.similarity_threshold})")
        result = await resolver.run()
        print("✓ Entity resolution with TF‑IDF completed.")
        print(result)
        return result


# -----------------------------------------------------------------------------
# 3.  __main__ guard -----------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"Entity resolution failed: {exc}")
        sys.exit(1)
