"""
database.py — Couche d'Accès aux Données SQLite pour le Pipeline P&L Multi-Agents

Strictement inspiré des notebooks Google ADK :

  Day 3a — Sessions (persistance & traçabilité)
  ├── §2.2  Session = conteneur, Events = briques de conversation
  ├── §2.4  InMemorySessionService  → prototypage rapide
  ├── §3.2  DatabaseSessionService  → sqlite:///my_agent_data.db
  ├── §3.6  Inspection directe     → SELECT … FROM events
  ├── §4    EventsCompactionConfig  → résumés automatiques
  └── §5    ToolContext.state       → lecture / écriture de state

  Day 3b — Memory (mémoire long-terme)
  ├── §3.1  InMemoryMemoryService   → MemoryService de dev
  ├── §4    add_session_to_memory() → ingestion session → mémoire
  ├── §5    load_memory / preload_memory → retrieval tools
  ├── §6.2  after_agent_callback    → sauvegarde automatique
  └── §7    Memory Consolidation    → extraction de faits-clés

Mapping vers nos tables :
  ┌─────────────────────┬────────────────────────────────────────────┐
  │ Concept ADK         │ Table SQLite                               │
  ├─────────────────────┼────────────────────────────────────────────┤
  │ Session             │ conversations                              │
  │ Event               │ messages                                   │
  │ EventsCompaction    │ summaries                                  │
  │ Memory Consolidation│ summaries                                  │
  │ after_agent_callback│ hitl_feedback                              │
  │ adk eval            │ evaluation_runs                            │
  └─────────────────────┴────────────────────────────────────────────┘

Contraintes :
  - AUCUNE logique métier — uniquement la couche d'accès aux données
  - Compatible Google Colab (chemin SQLite local, mono-thread tolérant)
  - Toutes les écritures sont atomiques (transaction + rollback)

Usage :
    >>> from database import DatabaseManager
    >>> db = DatabaseManager()                         # → pnl_analysis.db
    >>> cid = db.create_conversation("user_42")
    >>> db.log_message(cid, "A1_Normalization", "output", {"status": "ok"})
    >>> trail = db.get_audit_trail(cid)
    >>> db.close()
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

# ── Logger ───────────────────────────────────────────────────────
logger = logging.getLogger("pnl_database")


# ══════════════════════════════════════════════════════════════════
#  DatabaseManager
# ══════════════════════════════════════════════════════════════════
class DatabaseManager:
    """Couche d'accès aux données SQLite — zéro logique métier.

    Reproduit les mécanismes de persistance de Google ADK :

    ┌──────────────────────────────────────────────────────────────┐
    │  Day 3a §3.2 — DatabaseSessionService                       │
    │  → Connexion SQLite avec ``sqlite:///…`` pour persister     │
    │    les sessions au-delà des redémarrages du kernel.         │
    │                                                              │
    │  Day 3a §2.2 — Session.Events                               │
    │  → Chaque interaction (user input, agent response, tool     │
    │    call, tool output) est un Event chronologique.           │
    │                                                              │
    │  Day 3a §4 — EventsCompactionConfig                        │
    │  → Après N tours, un résumé remplace l'historique ancien.   │
    │                                                              │
    │  Day 3a §5 — ToolContext.state                              │
    │  → State {key: value} partagé entre outils et sous-agents. │
    │                                                              │
    │  Day 3b §4 — add_session_to_memory()                       │
    │  → Transfert de données de session vers la mémoire LT.     │
    │                                                              │
    │  Day 3b §6.2 — after_agent_callback                        │
    │  → Sauvegarde automatique après chaque tour d'agent.        │
    │                                                              │
    │  Day 3b §7 — Memory Consolidation                          │
    │  → Extraction de faits-clés depuis les événements bruts.    │
    └──────────────────────────────────────────────────────────────┘
    """

    # ──────────────────────────────────────────────────────────────
    #  CONSTRUCTION & INITIALISATION
    # ──────────────────────────────────────────────────────────────

    def __init__(self, db_path: str = "pnl_analysis.db") -> None:
        """Ouvre (ou crée) la base SQLite et initialise le schéma.

        Réf. Day 3a §3.2 :
            ``db_url = "sqlite:///my_agent_data.db"``
            ``session_service = DatabaseSessionService(db_url=db_url)``

        Sous Google Colab le CWD est ``/content``, donc le fichier
        sera créé à ``/content/pnl_analysis.db`` par défaut.

        Args:
            db_path: chemin vers le fichier .db (défaut: pnl_analysis.db).
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._init_schema()
        logger.info("DatabaseManager initialisé — %s", self.db_path)

    # ──────────────────────────────────────────────────────────────
    #  CONNEXION ROBUSTE
    #  Réf. Day 3a §3.2 : DatabaseSessionService crée la connexion
    #  SQLite automatiquement à l'initialisation.
    # ──────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        """Crée la connexion SQLite avec des réglages Colab-friendly.

        - check_same_thread=False : requis pour les notebooks async
        - timeout=30 : tolérance si verrouillage concurrent
        - WAL mode : meilleures performances en lecture concurrente
        - foreign_keys : intégrité référentielle activée
        - row_factory : accès aux colonnes par nom
        """
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row

    @property
    def conn(self) -> sqlite3.Connection:
        """Accès à la connexion avec reconnexion automatique si fermée."""
        if self._conn is None:
            self._connect()
            self._init_schema()
        return self._conn

    @contextmanager
    def _transaction(self):
        """Context-manager pour écritures atomiques avec rollback.

        Pattern identique à celui de DatabaseSessionService qui
        garantit la cohérence des Events persistés (Day 3a §3).
        """
        try:
            yield self.conn
            self.conn.commit()
        except sqlite3.Error as exc:
            self.conn.rollback()
            logger.error("SQLite transaction error: %s", exc)
            raise

    # ──────────────────────────────────────────────────────────────
    #  SCHÉMA
    # ──────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        """Crée les tables si elles n'existent pas encore.

        Réf. Day 3a §3.6 — le schéma ADK interne contient une table
        ``events`` avec les colonnes (app_name, session_id, author,
        content).  Nous reproduisons ce pattern avec nos propres
        tables adaptées au pipeline P&L multi-agents.
        """
        self.conn.executescript(
            """
            -- ═══════════════════════════════════════════════════════
            -- TABLE : conversations
            --
            -- ≈ Session ADK (Day 3a §2.2)
            -- Chaque exécution du pipeline P&L = 1 conversation.
            -- Comparable à :
            --   session = await session_service.create_session(
            --       app_name=APP_NAME, user_id=USER_ID,
            --       session_id=session_name
            --   )
            -- ═══════════════════════════════════════════════════════
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id         TEXT,
                status          TEXT DEFAULT 'initialized',
                metadata_json   TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_conv_user
                ON conversations (user_id);
            CREATE INDEX IF NOT EXISTS idx_conv_status
                ON conversations (status);

            -- ═══════════════════════════════════════════════════════
            -- TABLE : messages
            --
            -- ≈ Session.Events ADK (Day 3a §2.2)
            -- Chaque échange agent/outil = 1 message (sérialisé JSON).
            -- Comparable à la table interne ADK :
            --   SELECT app_name, session_id, author, content
            --   FROM events  (Day 3a §3.6)
            -- ═══════════════════════════════════════════════════════
            CREATE TABLE IF NOT EXISTS messages (
                message_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                agent_name      TEXT NOT NULL,
                role            TEXT NOT NULL,
                content_json    TEXT NOT NULL,
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations (conversation_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_msg_conv
                ON messages (conversation_id);
            CREATE INDEX IF NOT EXISTS idx_msg_agent
                ON messages (agent_name);

            -- ═══════════════════════════════════════════════════════
            -- TABLE : summaries
            --
            -- ≈ EventsCompactionConfig (Day 3a §4)
            --   Le Runner compacte l'historique après N tours en
            --   remplaçant les anciens Events par un résumé unique.
            --
            -- ≈ Memory Consolidation (Day 3b §7)
            --   Extraction de faits-clés depuis les échanges bruts.
            --   "User's favorite color: BlueGreen" au lieu de
            --   50 messages bruts.
            --
            -- Utilisé aussi pour re-alimenter load_memory /
            -- preload_memory lors d'une reprise (Day 3b §5).
            -- ═══════════════════════════════════════════════════════
            CREATE TABLE IF NOT EXISTS summaries (
                summary_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                summary_text    TEXT NOT NULL,
                agent_name      TEXT DEFAULT 'compactor',
                token_usage     INTEGER DEFAULT 0,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations (conversation_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sum_conv
                ON summaries (conversation_id);

            -- ═══════════════════════════════════════════════════════
            -- TABLE : evaluation_runs
            --
            -- Résultats des évaluations ``adk eval``.
            -- Stocke les métriques de chaque test-case.
            -- ═══════════════════════════════════════════════════════
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                run_id          TEXT PRIMARY KEY,
                conversation_id TEXT,
                test_case_id    TEXT,
                metric_name     TEXT,
                metric_value    REAL,
                passed          BOOLEAN,
                logs            TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations (conversation_id)
                    ON DELETE SET NULL
            );

            -- ═══════════════════════════════════════════════════════
            -- TABLE : hitl_feedback
            --
            -- ≈ after_agent_callback (Day 3b §6.2)
            --   Le callback sauvegarde automatiquement les données
            --   après chaque tour d'agent.  Ici, on enregistre les
            --   décisions humaines sur les anomalies (HITL).
            -- ═══════════════════════════════════════════════════════
            CREATE TABLE IF NOT EXISTS hitl_feedback (
                feedback_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                anomaly_id      TEXT NOT NULL,
                human_decision  TEXT NOT NULL,
                human_comment   TEXT DEFAULT '',
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations (conversation_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_hitl_conv
                ON hitl_feedback (conversation_id);

            -- ═══════════════════════════════════════════════════════
            -- TABLE : judge_history
            --
            -- Mémoire long-terme du A5_Quality_Judge.
            -- Persiste les constats, scores et recommandations
            -- de chaque exécution pour que les runs futurs puissent
            -- s'améliorer en s'appuyant sur l'historique.
            --
            -- Réf. Day 3b §7 — Memory Consolidation :
            --   Extraction de faits-clés depuis les évaluations
            --   passées pour alimenter le judge en contexte.
            -- ═══════════════════════════════════════════════════════
            CREATE TABLE IF NOT EXISTS judge_history (
                finding_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                run_timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                global_score    REAL,
                scores_json     TEXT NOT NULL,
                weaknesses_json TEXT NOT NULL,
                improvements_json TEXT NOT NULL,
                redundancies_json TEXT DEFAULT '[]',
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations (conversation_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_judge_conv
                ON judge_history (conversation_id);
            CREATE INDEX IF NOT EXISTS idx_judge_ts
                ON judge_history (run_timestamp);
            """
        )
        self.conn.commit()

    # ══════════════════════════════════════════════════════════════
    #  CONVERSATIONS  (≈ Sessions ADK — Day 3a §2.4, §3.2)
    # ══════════════════════════════════════════════════════════════

    def create_conversation(
        self,
        user_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crée une nouvelle conversation et retourne son UUID.

        Réf. Day 3a §2.4 :
            ``session = await session_service.create_session(
                app_name=app_name, user_id=USER_ID,
                session_id=session_name
            )``

        Args:
            user_id:  identifiant utilisateur.
            metadata: dict libre sérialisé en JSON.

        Returns:
            conversation_id (UUID4 string).
        """
        cid = str(uuid.uuid4())
        meta = json.dumps(metadata or {}, ensure_ascii=False)
        with self._transaction():
            self.conn.execute(
                """INSERT INTO conversations
                   (conversation_id, user_id, status, metadata_json)
                   VALUES (?, ?, 'initialized', ?)""",
                (cid, user_id, meta),
            )
        logger.info("Conversation créée : %s (user=%s)", cid[:12], user_id)
        return cid

    def update_status(self, conversation_id: str, status: str) -> None:
        """Met à jour le statut d'une conversation.

        Statuts : initialized → in_progress → completed | error.
        """
        with self._transaction():
            self.conn.execute(
                """UPDATE conversations
                   SET status = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE conversation_id = ?""",
                (status, conversation_id),
            )

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Récupère les métadonnées d'une conversation.

        Réf. Day 3a §3.3 :
            ``session = await session_service.get_session(
                app_name=app_name, user_id=USER_ID,
                session_id=session_name
            )``
        """
        row = self.conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_conversations(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Liste les conversations avec filtres optionnels.

        Réf. Day 3a §3.5 :
            Chaque session est isolée — on peut lister par user_id.
        """
        query = "SELECT * FROM conversations WHERE 1=1"
        params: list = []
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    # ══════════════════════════════════════════════════════════════
    #  MESSAGES  (≈ Session.Events — Day 3a §2.2, §3.6)
    # ══════════════════════════════════════════════════════════════

    def log_message(
        self,
        conversation_id: str,
        agent_name: str,
        role: str,
        content: Any,
    ) -> int:
        """Enregistre un échange (outil / LLM) sous forme JSON.

        Réf. Day 3a §3.6 — les Events sont stockés dans la table
        interne d'ADK avec (app_name, session_id, author, content).
        Notre table ``messages`` reproduit ce pattern :
            agent_name ↔ author
            content_json ↔ content

        Réf. Day 3a §2.2 — types d'Events :
            - User Input    → role='input'
            - Agent Response → role='output'
            - Tool Call     → role='tool_call'
            - Tool Output   → role='tool_output'
            - Error         → role='error'

        Args:
            conversation_id: UUID de la conversation.
            agent_name:      nom de l'agent (ex. A1_Normalization).
            role:            'input' | 'output' | 'tool_call' | 'error'.
            content:         dict / list / str sérialisé en JSON.

        Returns:
            message_id auto-incrémenté.
        """
        content_str = (
            json.dumps(content, ensure_ascii=False, default=str)
            if not isinstance(content, str)
            else content
        )
        with self._transaction():
            cursor = self.conn.execute(
                """INSERT INTO messages
                   (conversation_id, agent_name, role, content_json)
                   VALUES (?, ?, ?, ?)""",
                (conversation_id, agent_name, role, content_str),
            )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_messages(
        self,
        conversation_id: str,
        agent_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Récupère les messages d'une conversation.

        Réf. Day 3a §3.6 :
            ``cursor.execute(
                "select app_name, session_id, author, content
                 from events"
            )``

        Args:
            conversation_id: UUID cible.
            agent_name:      filtre optionnel.

        Returns:
            Liste de dicts avec content_json parsé en Python.
        """
        query = "SELECT * FROM messages WHERE conversation_id = ?"
        params: list = [conversation_id]
        if agent_name:
            query += " AND agent_name = ?"
            params.append(agent_name)
        query += " ORDER BY timestamp ASC"

        rows = self.conn.execute(query, params).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            try:
                d["content"] = json.loads(d["content_json"])
            except (json.JSONDecodeError, TypeError):
                d["content"] = d["content_json"]
            result.append(d)
        return result

    def get_last_successful_agent(self, conversation_id: str) -> Optional[str]:
        """Identifie le dernier agent ayant produit un output valide.

        Utile pour la reprise sur erreur du pipeline séquentiel
        A1 → A2 → A3 → A4 → A5.

        Returns:
            Nom de l'agent (ex. 'A2_Classification') ou None.
        """
        row = self.conn.execute(
            """SELECT agent_name FROM messages
               WHERE conversation_id = ?
                 AND role = 'output'
                 AND content_json NOT LIKE '%"status": "error"%'
               ORDER BY timestamp DESC LIMIT 1""",
            (conversation_id,),
        ).fetchone()
        return row["agent_name"] if row else None

    # ══════════════════════════════════════════════════════════════
    #  AUDIT TRAIL
    #  Réf. Day 3a §3.6 — inspection directe de la DB
    # ══════════════════════════════════════════════════════════════

    def get_audit_trail(self, conversation_id: str) -> List[Tuple]:
        """Piste d'audit chronologique brute.

        Réf. Day 3a §3.6 — ``check_data_in_db()`` inspecte
        directement la table events avec un SELECT.

        Format: [(agent_name, role, content_excerpt, timestamp), …]
        """
        return self.conn.execute(
            """SELECT agent_name, role,
                      substr(content_json, 1, 200) AS content_excerpt,
                      timestamp
               FROM messages
               WHERE conversation_id = ?
               ORDER BY timestamp ASC""",
            (conversation_id,),
        ).fetchall()

    # ══════════════════════════════════════════════════════════════
    #  SUMMARIES  (≈ EventsCompaction Day 3a §4
    #              + Memory Consolidation Day 3b §7)
    # ══════════════════════════════════════════════════════════════

    def save_summary(
        self,
        conversation_id: str,
        summary_text: str,
        agent_name: str = "compactor",
        token_usage: int = 0,
    ) -> int:
        """Persiste un résumé compacté de la conversation.

        Réf. Day 3a §4 — EventsCompactionConfig :
            Le Runner compacte l'historique après N tours.
            « Il ne supprime pas les anciens Events ; il les
            remplace par un unique Event contenant le résumé. »

        Réf. Day 3b §7 — Memory Consolidation :
            « Extracting ONLY important facts while discarding
            conversational noise. »
            Ex. Input:  50 messages bruts
                Output: "User's favorite color: BlueGreen"

        Ce résumé peut ensuite être rechargé via load_memory /
        preload_memory (Day 3b §5) pour donner du contexte
        long-terme à l'agent.

        Args:
            conversation_id: UUID de la conversation.
            summary_text:    texte du résumé.
            agent_name:      agent ayant généré le résumé.
            token_usage:     tokens consommés pour la génération.

        Returns:
            summary_id auto-incrémenté.
        """
        with self._transaction():
            cursor = self.conn.execute(
                """INSERT INTO summaries
                   (conversation_id, summary_text, agent_name, token_usage)
                   VALUES (?, ?, ?, ?)""",
                (conversation_id, summary_text, agent_name, token_usage),
            )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_summaries(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Récupère les résumés d'une conversation.

        Réf. Day 3a §4.3 — vérification de la compaction :
            ``for event in final_session.events:
                if event.actions and event.actions.compaction: …``
        """
        return [
            dict(r)
            for r in self.conn.execute(
                """SELECT * FROM summaries
                   WHERE conversation_id = ?
                   ORDER BY created_at DESC""",
                (conversation_id,),
            ).fetchall()
        ]

    # ══════════════════════════════════════════════════════════════
    #  HITL FEEDBACK  (≈ after_agent_callback — Day 3b §6.2)
    # ══════════════════════════════════════════════════════════════

    def log_hitl(
        self,
        conversation_id: str,
        anomaly_id: str,
        decision: str,
        comment: str = "",
    ) -> int:
        """Enregistre un feedback humain sur une anomalie.

        Réf. Day 3b §6.2 — after_agent_callback :
            ``async def auto_save_to_memory(callback_context):
                await callback_context._invocation_context
                    .memory_service.add_session_to_memory(
                        callback_context._invocation_context.session
                    )``
            Le callback sauvegarde automatiquement après chaque
            tour d'agent.  Ici, ce sont les décisions humaines
            (HITL) qui sont persistées de la même manière.

        Args:
            conversation_id: UUID de la conversation.
            anomaly_id:      identifiant de l'anomalie.
            decision:        'approuvé' | 'rejeté' | 'modifié'.
            comment:         commentaire libre de l'analyste.

        Returns:
            feedback_id auto-incrémenté.
        """
        with self._transaction():
            cursor = self.conn.execute(
                """INSERT INTO hitl_feedback
                   (conversation_id, anomaly_id, human_decision, human_comment)
                   VALUES (?, ?, ?, ?)""",
                (conversation_id, anomaly_id, decision, comment),
            )
        logger.info(
            "HITL feedback enregistré : anomaly=%s decision=%s",
            anomaly_id,
            decision,
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_hitl_feedback(
        self, conversation_id: str
    ) -> List[Dict[str, Any]]:
        """Récupère tous les feedbacks HITL d'une conversation."""
        return [
            dict(r)
            for r in self.conn.execute(
                """SELECT * FROM hitl_feedback
                   WHERE conversation_id = ?
                   ORDER BY timestamp ASC""",
                (conversation_id,),
            ).fetchall()
        ]

    # ══════════════════════════════════════════════════════════════
    #  EVALUATION RUNS  (adk eval)
    # ══════════════════════════════════════════════════════════════

    def log_evaluation(
        self,
        conversation_id: str,
        test_case_id: str,
        metric_name: str,
        metric_value: float,
        passed: bool,
        logs: str = "",
    ) -> str:
        """Enregistre le résultat d'un test d'évaluation.

        Args:
            conversation_id: UUID de la conversation évaluée.
            test_case_id:    identifiant du cas de test.
            metric_name:     nom de la métrique.
            metric_value:    valeur numérique.
            passed:          le test est-il réussi ?
            logs:            traces textuelles.

        Returns:
            run_id (UUID4).
        """
        run_id = str(uuid.uuid4())
        with self._transaction():
            self.conn.execute(
                """INSERT INTO evaluation_runs
                   (run_id, conversation_id, test_case_id,
                    metric_name, metric_value, passed, logs)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    conversation_id,
                    test_case_id,
                    metric_name,
                    metric_value,
                    passed,
                    logs,
                ),
            )
        return run_id

    # ══════════════════════════════════════════════════════════════
    #  REPRISE SUR ERREUR
    #  Réf. Day 3a §3.4 — reprendre une session après redémarrage
    #  du kernel en utilisant le même session_id.
    # ══════════════════════════════════════════════════════════════

    def get_resumable_state(
        self, conversation_id: str
    ) -> Dict[str, Any]:
        """Construit un état de reprise pour relancer le pipeline.

        Réf. Day 3a §3.4 :
            « Restart the kernel, run all previous cells EXCEPT
            the run_session in 3.3, then run with the SAME
            session ID.  Because the session is loaded from the
            database, the agent should still remember. »

        Retourne :
            - conversation_id
            - status
            - last_successful_agent
            - resume_from (prochain agent à exécuter)
            - agent_outputs (dict {agent_name: dernier output})
        """
        conv = self.get_conversation(conversation_id)
        if not conv:
            return {"error": f"Conversation {conversation_id} introuvable."}

        last_agent = self.get_last_successful_agent(conversation_id)

        pipeline_order = [
            "A1_Normalization",
            "A2_Classification",
            "A3_Variance",
            "A4_Anomaly",
            "A5_Reporting",
        ]

        resume_from = pipeline_order[0]
        if last_agent and last_agent in pipeline_order:
            idx = pipeline_order.index(last_agent)
            if idx + 1 < len(pipeline_order):
                resume_from = pipeline_order[idx + 1]
            else:
                resume_from = "completed"

        agent_outputs: Dict[str, Any] = {}
        for agent in pipeline_order:
            msgs = self.get_messages(conversation_id, agent_name=agent)
            outputs = [m for m in msgs if m["role"] == "output"]
            if outputs:
                agent_outputs[agent] = outputs[-1].get("content", {})

        return {
            "conversation_id": conversation_id,
            "status": conv.get("status", "unknown"),
            "last_successful_agent": last_agent,
            "resume_from": resume_from,
            "agent_outputs": agent_outputs,
        }

    def restore_session_state(
        self, conversation_id: str
    ) -> Dict[str, Any]:
        """Reconstruit le session state ADK depuis les outputs persistés.

        Réf. Day 3a §5 — ToolContext.state :
            ``tool_context.state["user:name"] = user_name``
            ``tool_context.state["user:country"] = country``
            Les tools lisent/écrivent dans session.state.

        Réf. Day 3b §4 — add_session_to_memory :
            On transfère les données de session vers la mémoire
            long-terme pour les retrouver via search_memory.

        Utilisable pour ré-injecter dans ``session.state`` :
            ``session = await session_service.create_session(
                …, state=db.restore_session_state(cid)
            )``
        """
        state: Dict[str, Any] = {"conversation_id": conversation_id}
        resumable = self.get_resumable_state(conversation_id)
        agent_outputs = resumable.get("agent_outputs", {})

        key_mapping = {
            "A1_Normalization": "a1_output",
            "A2_Classification": "a2_output",
            "A3_Variance": "a3_output",
            "A4_Anomaly": "a4_output",
            "A5_Reporting": "a5_report_data",
        }
        for agent, key in key_mapping.items():
            if agent in agent_outputs:
                state[key] = agent_outputs[agent]

        hitl = self.get_hitl_feedback(conversation_id)
        if hitl:
            last_hitl = hitl[-1]
            state["hitl_feedback"] = {
                "decision": last_hitl.get("human_decision", ""),
                "comment": last_hitl.get("human_comment", ""),
            }

        return state

    # ══════════════════════════════════════════════════════════════
    #  STATISTIQUES & DIAGNOSTIC
    # ══════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """Statistiques globales sur la base de données.

        Réf. Day 3a §3.6 — check_data_in_db() :
            Inspection directe de la DB pour diagnostic.
        """
        stats: Dict[str, Any] = {}
        for table in (
            "conversations",
            "messages",
            "summaries",
            "evaluation_runs",
            "hitl_feedback",
        ):
            row = self.conn.execute(
                f"SELECT COUNT(*) AS cnt FROM {table}"  # noqa: S608
            ).fetchone()
            stats[f"{table}_count"] = row["cnt"] if row else 0

        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS cnt "
            "FROM conversations GROUP BY status"
        ).fetchall()
        stats["conversations_by_status"] = {
            r["status"]: r["cnt"] for r in rows
        }

        return stats

    # ══════════════════════════════════════════════════════════════
    #  NETTOYAGE & FERMETURE
    #  Réf. Day 3a §Cleanup : suppression de la DB pour repartir
    #  à zéro (``os.remove("my_agent_data.db")``)
    # ══════════════════════════════════════════════════════════════

    def delete_conversation(self, conversation_id: str) -> None:
        """Supprime une conversation et toutes ses données associées.

        Les FK avec ON DELETE CASCADE suppriment automatiquement
        les messages, summaries et hitl_feedback liés.
        """
        with self._transaction():
            self.conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
        logger.info("Conversation supprimée : %s", conversation_id[:12])

    # ══════════════════════════════════════════════════════════════
    #  JUDGE HISTORY  (Mémoire long-terme A5_Quality_Judge)
    #  Réf. Day 3b §7 — Memory Consolidation
    # ══════════════════════════════════════════════════════════════

    def save_judge_finding(
        self,
        conversation_id: str,
        global_score: float,
        scores: Dict[str, Any],
        weaknesses: List[str],
        improvements: List[str],
        redundancies: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Persiste les constats du A5_Quality_Judge pour mémoire long-terme.

        Args:
            conversation_id: UUID de la conversation évaluée.
            global_score:    score global 1-10.
            scores:          dict des scores par critère.
            weaknesses:      liste des points faibles détectés.
            improvements:    liste des recommandations d'amélioration.
            redundancies:    liste des redondances détectées dans le rapport.

        Returns:
            finding_id auto-incrémenté.
        """
        with self._transaction():
            cursor = self.conn.execute(
                """INSERT INTO judge_history
                   (conversation_id, global_score, scores_json,
                    weaknesses_json, improvements_json, redundancies_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    conversation_id,
                    global_score,
                    json.dumps(scores, ensure_ascii=False),
                    json.dumps(weaknesses, ensure_ascii=False),
                    json.dumps(improvements, ensure_ascii=False),
                    json.dumps(redundancies or [], ensure_ascii=False),
                ),
            )
        logger.info(
            "Judge finding persisté : conv=%s score=%.1f",
            conversation_id[:12], global_score,
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_judge_history(
        self,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Récupère l'historique des évaluations du judge.

        Retourne les N dernières évaluations avec scores et constats
        pour que le judge puisse s'appuyer sur l'expérience passée.

        Args:
            limit: nombre maximal de résultats (défaut: 10).

        Returns:
            Liste de dicts avec champs parsés depuis JSON.
        """
        rows = self.conn.execute(
            """SELECT * FROM judge_history
               ORDER BY run_timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            for key in ("scores_json", "weaknesses_json",
                        "improvements_json", "redundancies_json"):
                try:
                    d[key.replace("_json", "")] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    d[key.replace("_json", "")] = d[key]
            result.append(d)
        return result

    def close(self) -> None:
        """Ferme proprement la connexion SQLite."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Connexion SQLite fermée.")

    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


# ══════════════════════════════════════════════════════════════════
#  SMOKE TEST  (exécution directe depuis le terminal)
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    with DatabaseManager() as db:
        # 1. Créer une conversation (≈ create_session Day 3a §2.4)
        cid = db.create_conversation(
            user_id="test_user", metadata={"env": "local"}
        )
        print(f"✅ Conversation créée : {cid[:12]}…")

        # 2. Mettre à jour le statut
        db.update_status(cid, "in_progress")

        # 3. Logger des messages (≈ Events Day 3a §2.2)
        mid = db.log_message(
            cid,
            "A1_Normalization",
            "output",
            {
                "status": "success",
                "data_summary": {
                    "budget_rows": 6,
                    "actual_rows": 6,
                    "mapping_rows": 6,
                },
            },
        )
        print(f"✅ Message enregistré : id={mid}")

        db.log_message(
            cid,
            "A4_Anomaly",
            "output",
            {
                "anomalies": [{"id": "AN-test", "type": "Suspect"}],
                "requires_human_validation": True,
                "global_confidence_score": 0.45,
            },
        )

        # 4. HITL feedback (≈ after_agent_callback Day 3b §6.2)
        db.log_hitl(
            cid, "AN-test", "approuvé", "RAS après vérification"
        )

        # 5. Summary (≈ Compaction Day 3a §4 / Consolidation Day 3b §7)
        sid = db.save_summary(
            cid,
            "Résumé : 6 comptes analysés, 2 anomalies détectées.",
            token_usage=150,
        )
        print(f"✅ Résumé enregistré : id={sid}")

        db.update_status(cid, "completed")

        # 6. Reprise sur erreur (≈ resume session Day 3a §3.4)
        resume = db.get_resumable_state(cid)
        print(f"\n📋 État de reprise :")
        print(f"   Dernier agent OK : {resume['last_successful_agent']}")
        print(f"   Reprendre depuis : {resume['resume_from']}")

        # 7. Restore session state (≈ ToolContext.state Day 3a §5)
        state = db.restore_session_state(cid)
        print(f"\n📋 Session state restauré :")
        print(f"   Clés : {list(state.keys())}")

        # 8. Audit trail (≈ check_data_in_db Day 3a §3.6)
        trail = db.get_audit_trail(cid)
        print(f"\n📋 Audit trail ({len(trail)} entrées) :")
        for row in trail:
            print(f"   [{row[3]}] {row[0]} ({row[1]}) → {row[2][:80]}…")

        # 9. Stats
        stats = db.get_stats()
        print(f"\n📊 Stats : {json.dumps(stats, indent=2)}")

        print("\n✅ Smoke test complet — database.py est fonctionnel.")
