import os
import pandas as pd

ENCODINGS = ["utf-8", "latin1", "cp1252", "iso-8859-1", "utf-16"]
ALLOWED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".parquet", ".ods"}

def load_dataset(path: str) -> pd.DataFrame:
    """
    Safely loads dataset files into a pandas DataFrame.
    Supports CSV, TSV, TXT, XLSX, XLS, JSON, Parquet, and ODS formats.
    """
    if not os.path.exists(path):
        raise ValueError("Dataset file does not exist.")

    file_size = os.path.getsize(path)
    if file_size == 0:
        raise ValueError("The uploaded file is empty (0 bytes).")

    path_lower = path.lower()
    df = None

    if path_lower.endswith((".csv", ".tsv", ".txt")):
        for sep in [",", "\t", ";", "|"]:
            for enc in ENCODINGS:
                try:
                    df_candidate = pd.read_csv(path, sep=sep, encoding=enc, on_bad_lines="skip")
                    if df_candidate.shape[1] > 1 or sep == ",":
                        df = df_candidate
                        break
                except Exception:
                    continue
            if df is not None and not df.empty:
                break

        if df is None:
            raise ValueError("Could not read CSV/TSV file with any known encoding or delimiter.")

    elif path_lower.endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except ImportError as exc:
            raise ValueError("Excel support (.xlsx) requires openpyxl.") from exc
        except Exception as e:
            raise ValueError(f"Could not read Excel file: {str(e)}")

    elif path_lower.endswith(".xls"):
        try:
            df = pd.read_excel(path, engine="xlrd")
        except ImportError as exc:
            raise ValueError("Excel support (.xls) requires xlrd.") from exc
        except Exception as e:
            raise ValueError(f"Could not read XLS file: {str(e)}")

    elif path_lower.endswith(".json"):
        try:
            df = pd.read_json(path)
        except ValueError:
            try:
                df = pd.read_json(path, lines=True)
            except Exception as e:
                raise ValueError(f"Could not read JSON file: {str(e)}")

    elif path_lower.endswith(".parquet"):
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            raise ValueError(f"Could not read Parquet file: {str(e)}")

    elif path_lower.endswith(".ods"):
        try:
            df = pd.read_excel(path, engine="odf")
        except ImportError as exc:
            raise ValueError("ODS support requires odfpy.") from exc
        except Exception as e:
            raise ValueError(f"Could not read ODS file: {str(e)}")
    else:
        raise ValueError(f"Unsupported file type. Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    if df is None or df.empty or df.shape[0] == 0:
        raise ValueError("The uploaded dataset contains no data rows.")

    if df.shape[1] == 0:
        raise ValueError("The uploaded dataset contains no columns.")

    return df