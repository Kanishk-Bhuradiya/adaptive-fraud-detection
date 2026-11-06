# Steps 30-31 - Poisson / quasi-Poisson / negative-binomial regression of weekly fraud counts, with change points.
# SELF-CONTAINED: loads only 4 columns, so it can run in a fresh session (it replaces any existing `df`; run it last or in its own session).
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

DATA_DIR = "/kaggle/input/competitions/ieee-fraud-detection"

# ---------- load only the 4 columns we need ----------
df = pd.read_csv(f"{DATA_DIR}/train_transaction.csv", usecols=["TransactionID", "TransactionDT", "isFraud", "V4"])
df = df.sort_values("TransactionDT").reset_index(drop=True)
df["week"] = ((df["TransactionDT"] / 86400 - 1) // 7).astype(int)
print("rows:", len(df), "| weeks:", df["week"].min(), "to", df["week"].max())

# ---------- weekly counts (week 0 = warm-up, skipped) ----------
d = df[df["week"] >= 1]
wk = d.groupby("week")["isFraud"].agg(n="size", frauds="sum").reset_index()
wk["log_n"] = np.log(wk["n"])                      # "exposure": more transactions -> more frauds, all else equal
wk["rate_%"] = wk["frauds"] / wk["n"] * 100

# ---------- Model 1: Poisson regression, fraud rate with a time trend ----------
pois = smf.glm("frauds ~ week", data=wk, family=sm.families.Poisson(), offset=wk["log_n"]).fit()
dispersion = pois.pearson_chi2 / pois.df_resid     # about 1 = Poisson is fine; well above 1 = overdispersed
quasi = smf.glm("frauds ~ week", data=wk, family=sm.families.Poisson(), offset=wk["log_n"]).fit(scale="X2")
nb = smf.negativebinomial("frauds ~ week", data=wk, offset=wk["log_n"]).fit(disp=0)

def rate_ratio(fit, name="week"):
    ci = np.exp(fit.conf_int().loc[name])
    return f"{np.exp(fit.params[name]):.4f}  (95% CI {ci[0]:.4f} to {ci[1]:.4f}, p={fit.pvalues[name]:.3f})"

print("\nWeekly fraud RATE multiplier per extra week (1.000 = no trend):")
print("  Poisson                :", rate_ratio(pois))
print("  Poisson, quasi (robust):", rate_ratio(quasi))
print("  Negative binomial      :", rate_ratio(nb))
print(f"Dispersion (Pearson chi2 / df): {dispersion:.2f}   | NB overdispersion alpha: {nb.params['alpha']:.4f}")

# ---------- Model 2: does the trend differ between V4 missing and V4 present? ----------
g = d.assign(grp=np.where(d["V4"].isna(), "V4 missing", "V4 present"))
gw = g.groupby(["week", "grp"])["isFraud"].agg(n="size", frauds="sum").reset_index()
gw["log_n"] = np.log(gw["n"])
joint = smf.glm("frauds ~ grp * week", data=gw, family=sm.families.Poisson(), offset=gw["log_n"]).fit(scale="X2")
inter = [k for k in joint.params.index if ":" in k][0]
print(f"\nDifference in weekly trend between the two groups (interaction): rate ratio {np.exp(joint.params[inter]):.4f}, p={joint.pvalues[inter]:.3f}")
for name, gg in gw.groupby("grp"):
    f = smf.glm("frauds ~ week", data=gg, family=sm.families.Poisson(), offset=gg["log_n"]).fit(scale="X2")
    print(f"  {name:11s} weekly rate multiplier:", rate_ratio(f))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(wk["week"], wk["rate_%"], "o", label="observed")
ax[0].plot(wk["week"], pois.fittedvalues / wk["n"] * 100, "-", label="Poisson trend")
ax[0].set_xlabel("week"); ax[0].set_ylabel("fraud rate (%)"); ax[0].set_title("All transactions"); ax[0].legend()
for name, gg in gw.groupby("grp"):
    ax[1].plot(gg["week"], gg["frauds"] / gg["n"] * 100, "o-", label=name)
ax[1].set_xlabel("week"); ax[1].set_ylabel("fraud rate (%)"); ax[1].set_title("By V4 group"); ax[1].legend()
plt.tight_layout(); plt.show()

# ================= Step 31: change points at weeks 4 and 13 (chosen after looking at the plots - exploratory) =================
wk["legit"] = wk["n"] - wk["frauds"]
wk["step4"]  = (wk["week"] >= 4).astype(int)
wk["step13"] = (wk["week"] >= 13).astype(int)
gw["step4"]  = (gw["week"] >= 4).astype(int)
gw["step13"] = (gw["week"] >= 13).astype(int)

def fit_glm(formula, data, use_offset=True):
    off = data["log_n"] if use_offset else None
    return smf.glm(formula, data=data, family=sm.families.Poisson(), offset=off).fit(scale="X2")   # robust to overdispersion

def show(title, fit):
    print(title)
    ci = np.exp(fit.conf_int())
    for k in fit.params.index:
        if k == "Intercept":
            continue
        print(f"   {k:8s} multiplier {np.exp(fit.params[k]):.3f}  (95% CI {ci.loc[k, 0]:.3f} to {ci.loc[k, 1]:.3f}, p={fit.pvalues[k]:.3f})")

show("Weekly number of FRAUDS ~ step at week 4 + week", fit_glm("frauds ~ step4 + week", wk, use_offset=False))
show("Weekly number of LEGITIMATE transactions ~ step at week 4 + week", fit_glm("legit ~ step4 + week", wk, use_offset=False))
show("\nFraud RATE, all transactions ~ step4 + step13 + week", fit_glm("frauds ~ step4 + step13 + week", wk))
for name, gg in gw.groupby("grp"):
    show(f"Fraud RATE, {name} ~ step4 + step13 + week", fit_glm("frauds ~ step4 + step13 + week", gg))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(wk["week"], wk["n"], "o-"); ax[0].set_title("Transactions per week"); ax[0].set_xlabel("week"); ax[0].set_ylabel("transactions")
ax[1].plot(wk["week"], wk["frauds"], "o-", color="C3"); ax[1].set_title("Frauds per week"); ax[1].set_xlabel("week"); ax[1].set_ylabel("frauds")
for a in ax:
    a.axvline(3.5, color="gray", ls="--"); a.axvline(12.5, color="gray", ls=":")
plt.tight_layout(); plt.show()
