import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Definir directorios
csv_file = 'datasets/Kaggle - 100 Million+ Steam Reviews/all_reviews.csv'
parquet_file = 'datasets/parquet/steam/reviews/reviews.parquet'
chunk_size = 100_000

# Crear directorio de salida si no existe
os.makedirs(os.path.dirname(parquet_file), exist_ok=True)

# Aplicando formato fijo para columnas (previene inferencia erronea)
csv_dtypes = {
    'recommendationid': 'Int64',
    'appid': 'Int64',
    'game': str,
    'author_steamid': 'Int64',
    'author_num_games_owned': 'Int64',
    'author_num_reviews': 'Int64',
    'author_playtime_forever': 'Int64',
    'author_playtime_last_two_weeks': 'Int64',
    'author_playtime_at_review': 'Int64',
    'author_last_played': 'Int64',
    'language': str,
    'review': str,
    'timestamp_created': 'Int64',
    'timestamp_updated': 'Int64',
    'voted_up': 'Int64',
    'votes_up': 'Int64',
    'votes_funny': 'Int64',
    'weighted_vote_score': float,
    'comment_count': 'Int64',
    'steam_purchase': 'Int64',
    'received_for_free': 'Int64',
    'written_during_early_access': 'Int64',
    'hidden_in_steam_china': 'Int64',
    'steam_china_location': str
}
# -------------------------------------------------------

parquet_writer = None
print(f"Starting complete schema-enforced conversion for: {csv_file}")

# Procesar en bloques
for i, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size, dtype=csv_dtypes, low_memory=False)):
    
    # Aplicar tipos de columnas
    for col in chunk.columns:
        if chunk[col].dtype == 'object':
            chunk[col] = chunk[col].astype(str)
            
    # Convertir a pyarrow
    table = pa.Table.from_pandas(chunk, preserve_index=False)
    
    # Preparar el Writer
    if parquet_writer is None:
        parquet_writer = pq.ParquetWriter(
            parquet_file, 
            table.schema, 
            compression='zstd'
        )
        print("Initialized Parquet file with a unified schema layout.")
        
    # Realizar append del bloque procesado
    parquet_writer.write_table(table)
    
    if (i + 1) % 10 == 0:
        print(f"Processed {(i + 1) * chunk_size:,} total rows...")

# Finalizar y cerrar el writer
if parquet_writer:
    parquet_writer.close()

print(f"\nConversion successfully completed: {parquet_file}")
print(f"Final Parquet File Size: {os.path.getsize(parquet_file) / (1024**3):.2f} GB")
