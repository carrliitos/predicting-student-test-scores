import pandas as pd
from sklearn.preprocessing import StandardScaler

def one_hot_encode(X_train: pd.DataFrame, 
                   X_val: pd.DataFrame, 
                   encode_cols: list[str], 
                   num_scale_cols: list[str], 
                   X_test: pd.DataFrame | None = None, 
                   drop_id_col: str | None = "id",):
  # Drop id
  if drop_id_col:
    X_train = X_train.drop(columns=[drop_id_col], errors="ignore")
    X_val = X_val.drop(columns=[drop_id_col], errors="ignore")
    if X_test is not None:
      X_test = X_test.drop(columns=[drop_id_col], errors="ignore")

  # One-hot encode with a reference level dropped
  X_train_enc = pd.get_dummies(X_train, columns=encode_cols, dtype=int, drop_first=True)
  X_val_enc = pd.get_dummies(X_val,   columns=encode_cols, dtype=int, drop_first=True)
  X_test_enc = None
  if X_test is not None:
    X_test_enc = pd.get_dummies(X_test, columns=encode_cols, dtype=int, drop_first=True)

  # Align columns so val/test match train exactly
  X_val_enc = X_val_enc.reindex(columns=X_train_enc.columns, fill_value=0)
  if X_test_enc is not None:
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

  # Drop constant columns
  const_cols = X_train_enc.columns[X_train_enc.std(axis=0) == 0]
  if len(const_cols) > 0:
    X_train_enc = X_train_enc.drop(columns=const_cols)
    X_val_enc = X_val_enc.drop(columns=const_cols, errors="ignore")
    if X_test_enc is not None:
      X_test_enc = X_test_enc.drop(columns=const_cols, errors="ignore")

  # Scale numeric columns
  scaler = StandardScaler()
  X_train_enc[num_scale_cols] = scaler.fit_transform(X_train_enc[num_scale_cols])
  X_val_enc[num_scale_cols] = scaler.transform(X_val_enc[num_scale_cols])
  if X_test_enc is not None:
    X_test_enc[num_scale_cols] = scaler.transform(X_test_enc[num_scale_cols])

  return X_train_enc, X_val_enc, X_test_enc, scaler, const_cols
