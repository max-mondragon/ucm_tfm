import argparse
import glob
import os
import pandas as pd

# Script para convertir archivos CSV a Parquet con compresión Zstandard (zstd)
# Validar la ruta de los archivos, este script se ejecutó en una imagen de Docker con el volumen montado en /UCM_TFM/datasets/...

def main():
    
    parser = argparse.ArgumentParser(
        description="Convierte archivos CSV a formato Parquet con compresión Zstandard (zstd)."
    )
    
    # Argumento para el directorio de entrada
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Patrón o ruta de los archivos CSV de entrada'
    )
    
    # Argumento para el directorio de salida
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Directorio de destino para guardar los archivos Parquet'
    )

    args = parser.parse_args()

    input_pattern = args.input
    if os.path.isdir(input_pattern):
        input_pattern = os.path.join(input_pattern, "*.csv")

    output_dir = args.output

    # Asegurar que el directorio de salida exista
    os.makedirs(output_dir, exist_ok=True)

    files = glob.glob(input_pattern)

    print(f"Found {len(files)} CSV files to convert.")

    for csv_path in files:
        try:
            # Obtener el nombre del archivo sin la extensión
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            
            # Construir la ruta de destino
            parquet_path = os.path.join(output_dir, f"{base_name}.parquet")
            
            print(f"Converting: {os.path.basename(csv_path)} -> {parquet_path}")
            
            # Leer y escribir el archivo
            df = pd.read_csv(csv_path)
            df.to_parquet(parquet_path, engine='pyarrow', compression='zstd', index=False)
            
        except Exception as e:
            print(f"Error processing {os.path.basename(csv_path)}: {e}")

    print("All file conversions completed.")

if __name__ == "__main__":
    main()