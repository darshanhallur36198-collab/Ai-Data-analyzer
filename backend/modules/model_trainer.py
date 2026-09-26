from backend.modules.ml_model import train_ml_model

def auto_train_model(df, target_col=None):
    """
    Backward-compatible wrapper for auto_train_model using the new ml_model module.
    """
    return train_ml_model(df, target_col=target_col)
