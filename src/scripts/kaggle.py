import os
import sys
import pandas as pd
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from pathlib import Path
from utils import logger
from utils import context
from utils import stat_funcs
from utils import preproc
from datetime import datetime

today = datetime.today().strftime("%Y-%m-%d")

directory = context.get_context(os.path.abspath(__file__))
logger_name = Path(__file__).stem
kaggle = logger.setup_logger(logger_name, f"{directory}\\logs\\main.log")
data_path = f"{directory}\\data"
version_num = "v03"

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

def add_linear_features(df: pd.DataFrame) -> pd.DataFrame:
  """
  Add minimal interaction / curvature terms that are friendly for linear models.
  Assumes numeric columns exist (possibly scaled) with these names.
  """
  required = {"study_hours",
              "sleep_hours", 
              "class_attendance", 
              "exam_difficulty_hard", 
              "exam_difficulty_moderate",
              "sleep_quality_good",
              "sleep_quality_poor",
              "study_method_mixed",
              "facility_rating_low",
              "facility_rating_medium"}
  missing = required - set(df.columns)
  if missing:
    raise KeyError(f"Missing required columns for linear features: {sorted(missing)}")

  return (
    df.assign(
      study_hours_attendance=lambda x: x["study_hours"] * x["class_attendance"],
      study_sleep=lambda x: x["study_hours"] * x["sleep_hours"],
      attendance_sleep=lambda x: x["class_attendance"] * x["sleep_hours"],

      study_hours_exam_difficulty__hard=lambda x: x["study_hours"] * x["exam_difficulty_hard"],
      study_hours_exam_difficulty__moderate=lambda x: x["study_hours"] * x["exam_difficulty_moderate"],

      study_hours_sleep_quality__good=lambda x: x["study_hours"] * x["sleep_quality_good"],
      study_hours_sleep_quality__poor=lambda x: x["study_hours"] * x["sleep_quality_poor"],

      study_method_mixed_facility_rating__low=lambda x: x["study_method_mixed"] * x["facility_rating_low"],
      study_method_mixed_facility_rating__medium=lambda x: x["study_method_mixed"] * x["facility_rating_medium"],

      study_method_mixed_exam_difficulty__hard=lambda x: x["study_method_mixed"] * x["exam_difficulty_hard"],
      study_method_mixed_exam_difficulty__moderate=lambda x: x["study_method_mixed"] * x["exam_difficulty_moderate"],

      study_hours_curve=lambda x: x["study_hours"] ** 2,
      sleep_hours_curve=lambda x: x["sleep_hours"] ** 2
    )
  )

def go():
  try:
    train_df = pd.read_csv(f"{directory}/data-raw/train.csv")
    test_df = pd.read_csv(f"{directory}/data-raw/test.csv")
    kaggle.info("Datasets loaded!")

    kaggle.info("===Feature Engineering===")
    TARGET = "exam_score"

    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    encode_cols = ["gender", "course", "internet_access", "study_method", "sleep_quality", "facility_rating", "exam_difficulty"]
    num_scale_cols = ["age", "study_hours", "class_attendance", "sleep_hours"]
    X_train_enc, X_val_enc, X_test_enc, scaler, dropped_const_cols = preproc.one_hot_encode(
      X_train=X_train,
      X_val=X_val,
      X_test=test_df,
      encode_cols=encode_cols,
      num_scale_cols=num_scale_cols,
      drop_id_col="id"
    )

    X_train_enc = add_linear_features(X_train_enc)
    X_val_enc = add_linear_features(X_val_enc)
    if X_test_enc is not None:
      X_test_enc = add_linear_features(X_test_enc)

    feature_columns = X_train_enc.columns.tolist()

    cond = np.linalg.cond(X_train_enc.to_numpy(float))
    kaggle.info(f"cond: {cond}")
    kaggle.info(f"dropped constant cols: {len(dropped_const_cols)}")
    kaggle.info("Encoded datasets built successfully!")
    kaggle.info(f"X_train_enc: {X_train_enc.shape} | X_val_enc: {X_val_enc.shape}")
    kaggle.info(f"y_train: {y_train.shape} | y_val: {y_val.shape}")
    kaggle.info(f"X_test_enc: {None if X_test_enc is None else X_test_enc.shape}")

    kaggle.info("===Modeling + Kaggle Submission===")
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    lr = LinearRegression()
    lr.fit(X_train_enc, y_train)
    val_pred = lr.predict(X_val_enc)

    mse = mean_squared_error(y_val, val_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, val_pred)
    r2 = r2_score(y_val, val_pred)
    kaggle.info(f"MSE : {mse:.4f}")
    kaggle.info(f"RMSE: {rmse:.4f}")
    kaggle.info(f"MAE : {mae:.4f}")
    kaggle.info(f"R^2 : {r2:.4f}")

    X_test_enc = X_test_enc[feature_columns]
    test_pred = lr.predict(X_test_enc)

    submission = pd.DataFrame({"id": test_df["id"], "exam_score": test_pred})
    submission_path = Path(f"{directory}/data/{version_num}-submission.csv")
    submission.to_csv(submission_path, index=False)
    kaggle.info(f"Saved: {submission_path.resolve()}")
  except ConnectionError as connection_error:
    kaggle.error(f"Unable to connect to db: {connection_error}")
    sys.exit(1)
  except KeyError as key_error:
    kaggle.error(f"Incorrect connection keys: {key_error}")
    sys.exit(1)

  kaggle.debug("Project complete.")
