"""
Servicio de recomendaciones ALS - Steam Games
Cloud Run + FastAPI

Endpoints:
  GET  /health              -> chequeo de salud / estado de carga
  POST /recommend/user      -> recomendaciones para un usuario conocido (via BigQuery user_vectors, o Firestore a futuro)
  POST /recommend/items     -> recomendaciones "cold start" a partir de una lista de item_ids (promedio de vectores)
"""

import os
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google.cloud import storage
from google.cloud import bigquery
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recommender")

# --------------------------------------------------------------------------
# Configuración (via variables de entorno, con defaults para desarrollo)
# --------------------------------------------------------------------------
GCS_BUCKET = os.environ.get("GCS_BUCKET", "ucm_tfm_recommendation_engine_max")
GCS_ITEM_FACTORS_PATH = os.environ.get("GCS_ITEM_FACTORS_PATH", "artifacts/item_factors.npz")
BQ_PROJECT = os.environ.get("BQ_PROJECT", "working-area-max")
BQ_ITEM_LOOKUP_TABLE = os.environ.get("BQ_ITEM_LOOKUP_TABLE", "working-area-max.ucm_master.item_lookup")
BQ_USER_FACTORS_TABLE = os.environ.get("BQ_USER_FACTORS_TABLE", "working-area-max.model_artifacts.user_factors")
STEAM_API_KEY = os.environ.get("STEAM_API_KEY", "")

# Lista de AppIDs a excluir manualmente de las recomendaciones (separados por coma).
# Útil para casos donde steam_games clasifica software/herramientas como juego válido
# (ej. 431960 = Wallpaper Engine). No requiere tocar item_factors.npz ni BigQuery.
EXCLUDED_APPIDS = {
    int(x.strip())
    for x in os.environ.get("EXCLUDED_APPIDS", "431960").split(",")
    if x.strip().isdigit()
}

# --------------------------------------------------------------------------
# Estado global cargado una sola vez al iniciar la instancia (cold start)
# --------------------------------------------------------------------------
STATE = {}


def load_item_factors():
    """Descarga item_factors.npz desde GCS y lo carga en memoria."""
    logger.info(f"Descargando gs://{GCS_BUCKET}/{GCS_ITEM_FACTORS_PATH} ...")
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    tmp_path = "/tmp/item_factors.npz"
    bucket.blob(GCS_ITEM_FACTORS_PATH).download_to_filename(tmp_path)

    data = np.load(tmp_path, allow_pickle=True)
    item_factors = data["item_factors"].astype(np.float32)
    item_ids = data["item_ids"]
    is_valid_game = data["is_valid_game"].astype(bool)

    logger.info(f"item_factors cargado: {item_factors.shape}, válidos: {is_valid_game.sum()}/{len(is_valid_game)}")
    return item_factors, item_ids, is_valid_game


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: se ejecuta una sola vez por instancia de Cloud Run ---
    item_factors, item_ids, is_valid_game = load_item_factors()

    STATE["item_factors"] = item_factors
    STATE["item_ids"] = item_ids
    id_to_idx = {int(v): i for i, v in enumerate(item_ids)}
    STATE["id_to_idx"] = id_to_idx

    # Índices inválidos = DLC/herramientas (is_valid_game=False) + exclusiones manuales
    invalid_from_flag = set(np.where(~is_valid_game)[0].tolist())
    invalid_from_manual = {id_to_idx[appid] for appid in EXCLUDED_APPIDS if appid in id_to_idx}
    STATE["invalid_idx"] = np.array(sorted(invalid_from_flag | invalid_from_manual), dtype=np.int64)

    if invalid_from_manual:
        logger.info(f"Exclusiones manuales aplicadas: {EXCLUDED_APPIDS & set(id_to_idx.keys())}")

    # Vectores normalizados, usados para el promedio en cold start
    norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8  # evitar división por cero en items sin interacciones
    STATE["norm_items"] = item_factors / norms

    STATE["bq_client"] = bigquery.Client(project=BQ_PROJECT)

    logger.info("Servicio listo.")
    yield
    # --- Shutdown: nada que limpiar explícitamente ---


app = FastAPI(title="Steam ALS Recommender", lifespan=lifespan)

# CORS abierto para la PoC (restringir en producción al dominio del bucket estático)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Modelos de request/response
# --------------------------------------------------------------------------
class ByUserRequest(BaseModel):
    user_id: str = Field(..., description="SteamID64 del usuario")
    n: int = Field(16, ge=1, le=100)


class ByItemsRequest(BaseModel):
    item_ids: list[int] = Field(..., min_length=1, description="AppIDs de Steam que le gustan al usuario")
    n: int = Field(16, ge=1, le=100)


class GameSearchResult(BaseModel):
    item_id: int
    name: str
    header_image: str | None = None


class GameSearchResponse(BaseModel):
    results: list[GameSearchResult]


class RecommendationItem(BaseModel):
    item_id: int
    score: float
    name: str | None = None
    header_image: str | None = None
    genres: str | None = None


class RecommendResponse(BaseModel):
    recommendations: list[RecommendationItem]
    coverage: dict | None = None


# --------------------------------------------------------------------------
# Lógica de scoring
# --------------------------------------------------------------------------
def compute_top_n(user_vector: np.ndarray, exclude_idx: set[int], n: int) -> list[tuple[int, float]]:
    """Calcula el top-N de items para un vector de usuario dado."""
    item_factors = STATE["item_factors"]
    item_ids = STATE["item_ids"]
    invalid_idx = STATE["invalid_idx"]

    scores = item_factors @ user_vector

    # Excluir DLC / herramientas / software (siempre)
    scores[invalid_idx] = -np.inf

    # Excluir items ya conocidos por el usuario (historial o input de cold start)
    if exclude_idx:
        idx_array = np.fromiter(exclude_idx, dtype=np.int64)
        idx_array = idx_array[idx_array < len(scores)]
        scores[idx_array] = -np.inf

    n = min(n, len(scores))
    top_idx = np.argpartition(-scores, n - 1)[:n]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    return [(int(item_ids[i]), float(scores[i])) for i in top_idx]


def search_games_by_name(query_text: str, limit: int = 10) -> list[dict]:
    """Busca juegos por coincidencia parcial de nombre, solo entre juegos válidos (no DLC/tools)."""
    bq_client = STATE["bq_client"]
    excluded = list(EXCLUDED_APPIDS) or [-1]  # -1 evita un UNNEST vacío si la lista quedó vacía
    sql = f"""
        SELECT item_id, game_name, header_image
        FROM `{BQ_ITEM_LOOKUP_TABLE}`
        WHERE is_valid_game = TRUE
          AND item_id NOT IN UNNEST(@excluded)
          AND LOWER(game_name) LIKE CONCAT('%', LOWER(@q), '%')
        ORDER BY positive_reviews DESC
        LIMIT @limit
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("q", "STRING", query_text),
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
            bigquery.ArrayQueryParameter("excluded", "INT64", excluded),
        ]
    )
    rows = bq_client.query(sql, job_config=job_config).result()
    return [
        {"item_id": row.item_id, "name": row.game_name, "header_image": row.header_image}
        for row in rows
    ]


def fetch_metadata(item_ids: list[int]) -> dict[int, dict]:
    """Consulta metadata (nombre, imagen, género) desde BigQuery para una lista de item_ids."""
    if not item_ids:
        return {}

    bq_client = STATE["bq_client"]
    query = f"""
        SELECT item_id, game_name, header_image, genres
        FROM `{BQ_ITEM_LOOKUP_TABLE}`
        WHERE item_id IN UNNEST(@ids)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("ids", "INT64", item_ids)]
    )
    rows = bq_client.query(query, job_config=job_config).result()

    return {
        row.item_id: {
            "name": row.game_name,
            "header_image": row.header_image,
            "genres": row.genres,
        }
        for row in rows
    }


def fetch_user_vector(user_id: str) -> np.ndarray | None:
    """Consulta el vector de un usuario conocido desde BigQuery (user_factors)."""
    bq_client = STATE["bq_client"]
    query = f"""
        SELECT user_factors
        FROM `{BQ_USER_FACTORS_TABLE}`
        WHERE user_id = @user_id
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("user_id", "STRING", user_id)]
    )
    rows = list(bq_client.query(query, job_config=job_config).result())
    if not rows:
        return None

    raw = rows[0].user_factors
    return np.array(raw, dtype=np.float32)


def build_response(recs: list[tuple[int, float]], coverage: dict | None = None) -> RecommendResponse:
    """Enriquecer resultados de scoring con metadata de BigQuery."""
    ids = [r[0] for r in recs]
    metadata = fetch_metadata(ids)

    items = []
    for item_id, score in recs:
        meta = metadata.get(item_id, {})
        items.append(
            RecommendationItem(
                item_id=item_id,
                score=score,
                name=meta.get("name"),
                header_image=meta.get("header_image"),
                genres=meta.get("genres"),
            )
        )
    return RecommendResponse(recommendations=items, coverage=coverage)


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "items_loaded": int(STATE["item_factors"].shape[0]),
        "valid_games": int(len(STATE["item_factors"]) - len(STATE["invalid_idx"])),
    }


@app.get("/search/games", response_model=GameSearchResponse)
def search_games(q: str, limit: int = 10):
    if len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail="La búsqueda requiere al menos 2 caracteres.")
    limit = max(1, min(limit, 25))
    results = search_games_by_name(q.strip(), limit)
    return GameSearchResponse(results=[GameSearchResult(**r) for r in results])


@app.post("/recommend/user", response_model=RecommendResponse)
def recommend_by_user(req: ByUserRequest):
    user_vector = fetch_user_vector(req.user_id)
    if user_vector is None:
        raise HTTPException(
            status_code=404,
            detail="Usuario no encontrado en la base entrenada. Usar /recommend/items para cold start.",
        )

    recs = compute_top_n(user_vector, exclude_idx=set(), n=req.n)
    return build_response(recs)


@app.post("/recommend/items", response_model=RecommendResponse)
def recommend_by_items(req: ByItemsRequest):
    id_to_idx = STATE["id_to_idx"]
    norm_items = STATE["norm_items"]

    idxs = [id_to_idx[i] for i in req.item_ids if i in id_to_idx]
    if not idxs:
        raise HTTPException(status_code=400, detail="Ninguno de los item_ids proporcionados es válido.")

    synthetic_vector = norm_items[idxs].mean(axis=0)
    norm = np.linalg.norm(synthetic_vector)
    if norm > 0:
        synthetic_vector = synthetic_vector / norm

    recs = compute_top_n(synthetic_vector, exclude_idx=set(idxs), n=req.n)
    return build_response(recs)