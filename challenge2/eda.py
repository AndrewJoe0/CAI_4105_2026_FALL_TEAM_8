"""
eda.py  -  exploratory data analysis (training data only). Run:  python eda.py
Prints the key facts and saves three figures in figures/.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import DATA_DIR, ROOT

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
train = pd.read_csv(DATA_DIR / "challenge2_train.csv")
test = pd.read_csv(DATA_DIR / "challenge2_test.csv")

print("Train:", train.shape, " Test:", test.shape)
print("Missing values  train:", int(train.isna().sum().sum()), " test:", int(test.isna().sum().sum()))
print("Duplicate sessions (ignoring ID):", int(train.drop(columns='ID').duplicated().sum()))
rate = train["Revenue"].mean()
print(f"Purchases: {train['Revenue'].sum()} of {len(train)} = {rate:.1%}  -> imbalanced, so accuracy is misleading")
print(f"A model that always predicts 'no purchase' has accuracy {1 - rate:.1%} but F1 = 0")

# --- Figure 1: class balance -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4, 3.2))
counts = train["Revenue"].value_counts().reindex([False, True])
ax.bar(["No purchase", "Purchase"], counts.values, color=["#9aa5b1", "#c8461a"])
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v}\n({v / len(train):.1%})", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, counts.max() * 1.18); ax.set_ylabel("Sessions"); ax.set_title("Target is imbalanced")
fig.tight_layout(); fig.savefig(FIG / "class_balance.png", dpi=150); plt.close(fig)

# --- Figure 2: PageValues is the strongest signal --------------------------------------
share = (train["PageValues"] > 0).groupby(train["Revenue"]).mean()
print("\nShare of sessions with PageValues > 0:  no purchase "
      f"{share[False]:.1%},  purchase {share[True]:.1%}")
fig, ax = plt.subplots(figsize=(5, 3.2))
for flag, color, label in [(False, "#9aa5b1", "No purchase"), (True, "#c8461a", "Purchase")]:
    ax.hist(np.log1p(train.loc[(train.Revenue == flag) & (train.PageValues > 0), "PageValues"]),
            bins=30, alpha=.75, color=color, label=label, density=True)
ax.set_xlabel("log(1 + PageValues), sessions with PageValues > 0 only"); ax.set_ylabel("Density")
ax.legend(); ax.set_title("PageValues vs. purchase"); fig.tight_layout()
fig.savefig(FIG / "pagevalues.png", dpi=150); plt.close(fig)

# --- Figure 3: purchase rate by month and visitor type ------------------------------------
print("\nPurchase rate by VisitorType:\n", train.groupby("VisitorType")["Revenue"].agg(["mean", "count"]).round(3))
order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
by_month = train.groupby("Month")["Revenue"].agg(["mean", "count"]).reindex(order)
print("\nPurchase rate by Month:\n", by_month.round(3))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [2.2, 1]})
axes[0].bar(order, by_month["mean"], color="#1f5fb0"); axes[0].axhline(rate, ls="--", color="k", lw=1)
axes[0].set_ylabel("Purchase rate"); axes[0].set_title("By month (dashed = overall)")
vt = train.groupby("VisitorType")["Revenue"].mean()
axes[1].bar(["New", "Other", "Returning"], vt.values, color="#1f5fb0"); axes[1].set_title("By visitor type")
fig.tight_layout(); fig.savefig(FIG / "month_visitor.png", dpi=150); plt.close(fig)

# --- Train vs test: are they similar? -------------------------------------------------
print("\nTrain vs test mean of key features (should be similar):")
cols = ["PageValues", "BounceRates", "ExitRates", "ProductRelated", "ProductRelated_Duration"]
print(pd.DataFrame({"train": train[cols].mean(), "test": test[cols].mean()}).round(3))
print("\nCategories that appear only in the test set:",
      {c: sorted(set(test[c]) - set(train[c])) for c in ["Month", "VisitorType", "Browser", "Region", "TrafficType", "OperatingSystems"]})
