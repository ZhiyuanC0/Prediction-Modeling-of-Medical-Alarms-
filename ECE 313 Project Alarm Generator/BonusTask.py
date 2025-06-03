#!/usr/bin/env python3
"""
ECE 313 - G  ▸  Bonus Task  (STUDENT STARTER CODE)
================================================================
    Fill in every "TODO" and run:
    $ python bonus_skeleton.py  --train-patients "1 2 ..." --test-patients "..."
"""
from __future__ import annotations
import argparse, pathlib, random, warnings
from typing import Tuple, Dict, List

import numpy as np
from scipy.io import loadmat
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression

# ────────────────────────────────────────────────────────────────────────────
#                       ░ U T I L I T I E S  ░
# ────────────────────────────────────────────────────────────────────────────
def load_patient(mat_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y)."""
    data = loadmat(mat_path)
    X = data["all_data"].astype(np.float32)
    y = data["all_labels"].ravel().astype(np.int64)
    return X, y


def list_patient_files(folder: str = ".") -> Dict[int, str]:
    """Map patient index → file path."""
    return {
        int(p.name.split("_")[0]): str(p)
        for p in pathlib.Path(folder).glob("*.mat")
    }


def split_train_test(patient_files: Dict[int, str],
                     train_ids: List[int], test_ids: List[int]
                     ) -> Tuple[np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray]:
    # TODO ── implement (see hand-out Bonus Task 0)
    train_labels = []
    test_labels = []

    train_data = []
    test_data = []

    for feature in range(7): #
        feature_arr_train = []
        feature_arr_test = []
        for pid in train_ids:
            X, y = load_patient(patient_files[pid])
            feature_arr_train.extend(X[feature])
        train_data.append(feature_arr_train)
        for pid in test_ids:
            X, y = load_patient(patient_files[pid])
            feature_arr_test.extend(X[feature])
        test_data.append(feature_arr_test)

    for pid in train_ids:
        X, y = load_patient(patient_files[pid])
        train_labels.extend(y)
    for pid in test_ids:
        X, y = load_patient(patient_files[pid])
        test_labels.extend(y)

    return (
        np.array(train_data),
        np.array(train_labels),
        np.array(test_data),
        np.array(test_labels)
        )


def normalise(train_x: np.ndarray, test_x: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
    # TODO ── implement
    for i in range(7):
        mean = train_x[i].mean(axis=0)
        std = train_x[i].std(axis=0)
        #std[std == 0] = 1.0  # prevent division by zero
        train_x[i] = (train_x[i] - mean)/std
        test_x[i] = (test_x[i] - mean)/std
    return (train_x,test_x)

def empirical_priors(y_train: np.ndarray) -> Tuple[float, float]:
    # TODO ── implement
    p1 = np.mean(y_train == 1)
    p0 = 1-p1
    return (p0, p1)

def metrics(y_true: np.ndarray, y_pred: np.ndarray
           ) -> Dict[str, float]:
   # TODO ── implement
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))  # false alarm
    fn = np.sum((y_true == 1) & (y_pred == 0))  # missed detection
    total = len(y_true)

    return {
        "false_alarm": fp / (fp + tn + 1e-10),         # FAR = FP / (FP + TN)
        "miss_detection": fn / (fn + tp + 1e-10),      # MDR = FN / (FN + TP)
        "error_rate": (fp + fn) / total                # ERR = (FP + FN) / total
    }



# ────────────────────────────────────────────────────────────────────────────
#                ░  L O G I S T I C   R E G R E S S I O N  ░
# ────────────────────────────────────────────────────────────────────────────
def train_logistic_regression(X_tr: np.ndarray, y_tr: np.ndarray):
    """Return a fitted sklearn.linear_model.LogisticRegression."""
    # TODO ── implement
    classifier = LogisticRegression(solver="liblinear")
    classifier.fit(X_tr, y_tr)
    return classifier



# ────────────────────────────────────────────────────────────────────────────
#                  ░  F E E D - F O R W A R D   N N  ░
# ────────────────────────────────────────────────────────────────────────────
class FeedForwardNN(nn.Module):
    def __init__(self, d_in: int, d_h: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(),
            nn.Linear(d_h, d_h),   nn.ReLU(),
            nn.Linear(d_h, 1),     nn.Sigmoid())
    def forward(self, x):          # (B,d) → (B,)
        return self.net(x).squeeze(1)


def train_nn(x_tr, y_tr, x_val, y_val,
             *, epochs=100, lr=1e-3, batch=256, patience=15, seed=0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FeedForwardNN(x_tr.shape[1]).to(dev).train()
    opt = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    best, best_val, wait = None, 1e9, 0
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.float().to(dev)
            opt.zero_grad(); loss_fn(net(xb), yb).backward(); opt.step()
        with torch.no_grad():
            v = loss_fn(net(torch.from_numpy(x_val).to(dev)),
                        torch.from_numpy(y_val).float().to(dev)).item()
        if v < best_val: best, best_val, wait = net.state_dict(), v, 0
        else:
            wait += 1
            if wait >= patience: break
    net.load_state_dict(best); net.eval().cpu(); return net


# ────────────────────────────────────────────────────────────────────────────
#                  ░       M A I N     ░
# ────────────────────────────────────────────────────────────────────────────
def main(args):
    pat_files = list_patient_files(args.data_dir)
    train_ids = [int(i) for i in args.train_patients.split()]
    test_ids  = [int(i) for i in args.test_patients.split()]

    # 1) load & normalise -----------------------------------------------------
    # TODO – call split_train_test, normalise, empirical_priors
    data_tuple = split_train_test(pat_files,train_ids,test_ids)
    (norm_train, norm_test) = normalise(data_tuple[0],data_tuple[2])
    (p0,p1) = empirical_priors(data_tuple[1])#load the training labels

    # 2) logistic-regression --------------------------------------------------
    # TODO – fit, predict probabilities, evaluate under τ_ML & τ_MAP
    τ_ML = 0.5
    τ_MAP = p0/(p0+p1)
    lm = train_logistic_regression(norm_train.T, data_tuple[1]) # logistic model
    lp = lm.predict_proba(norm_test.T)[:, 1] # logistic probabilities
    lr_results = {
        "logistic: tau=0.5": metrics(data_tuple[3], (lp >= τ_ML).astype(int)),
        "logistic: tau=MAP": metrics(data_tuple[3], (lp >= τ_MAP).astype(int))
    }
    print(lr_results)

    # 3) neural-network -------------------------------------------------------
    # TODO – evaluate same as above

    # Below is a sample implementation of the neural network training and
    # evaluation. You can modify it as per your requirements.
    # neural network (20 % validation split)
    # msk = np.random.rand(len(ytr)) < 0.8
    # nn  = train_nn(Xtr[msk], ytr[msk], Xtr[~msk], ytr[~msk],
    #                epochs=a.nn_epochs, seed=a.seed)
    # with torch.no_grad():
    #     nn_scores = nn(torch.from_numpy(Xte)).numpy()

    # def eval(scores, tag):
    #     return {
    #         f"{tag}_tau0.5": metrics(yte, (scores >= τ_ML ).astype(int)),
    #         f"{tag}_tauMAP": metrics(yte, (scores >= τ_MAP).astype(int))
    #     }

    # 4) save CSVs + optional figures  (Bonus Task 2) -------------------------
    # TODO – write global_bonus_metrics.csv + per-patient csv
    msk = np.random.rand(len(data_tuple[1])) < 0.8
    x_tr = norm_train.T[msk]
    y_tr = data_tuple[1][msk]
    x_val = norm_train.T[~msk]
    y_val = data_tuple[1][~msk]
    nn = train_nn(x_tr,y_tr, x_val, y_val)
    with torch.no_grad():
        nn_scores = nn(torch.from_numpy(norm_test.T)).numpy()
   
    nn_results = {
        "nn_tau0.5": metrics(data_tuple[3], (nn_scores >= τ_ML ).astype(int)),
        "nn_tauMAP": metrics(data_tuple[3], (nn_scores >= τ_MAP).astype(int))
    }
    print(nn_results)
    pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--train-patients", required=True,
                    help="e.g. '1 2 4 5 6 7'")
    ap.add_argument("--test-patients",  required=True,
                    help="e.g. '3 8 9'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args)
