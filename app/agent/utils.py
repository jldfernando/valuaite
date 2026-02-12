import numpy as np
import pandas as pd
from typing import Any

def sanitize_value(value: Any) -> Any:
    """
    Recursively converts NumPy/Pandas types to standard Python types.
    Ensures the state is serializable for LangGraph checkpoints.
    """
    # 1. Handle NumPy scalars
    if isinstance(value, (np.float64, np.float32, np.float16)):
        return float(value)
    if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [sanitize_value(v) for v in value.tolist()]
    
    # 2. Handle Pandas types
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return [sanitize_value(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return value.fillna(0).to_dict(orient='records')
    
    # 3. Handle Python Collections
    if isinstance(value, dict):
        # Convert keys to string as well (prevents OPT_NON_STR_KEYS errors)
        return {str(k): sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_value(v) for v in value]
    
    # 4. Fallback for unexpected math types that might have __float__
    if hasattr(value, '__float__') and not isinstance(value, (float, int, str)):
        try:
            return float(value)
        except:
            pass
            
    return value

def sanitize_state(state: dict) -> dict:
    """Cleans an entire dictionary."""
    return {k: sanitize_value(v) for k, v in state.items()}
