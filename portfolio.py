# portfolio_qaoa.py

import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram

# -------------------------------
# NIFTY BANK TICKERS
# -------------------------------
nifty_bank_tickers = {
    "AXISBANK": "AXISBANK.NS",
    "BANDHANBNK": "BANDHANBNK.NS",
    "BANKBARODA": "BANKBARODA.NS",
    "FEDERALBNK": "FEDERALBNK.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "IDFCFIRSTB": "IDFCFIRSTB.NS",
    "INDUSINDBK": "INDUSINDBK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "PNB": "PNB.NS",
    "SBIN": "SBIN.NS",
    "RBLBANK": "RBLBANK.NS"
}

# -------------------------------
# FUNCTIONS: NSE μ/Σ
# -------------------------------
def get_data_selected(selected, start, end):
    df_list = []
    for name in selected:
        ticker = nifty_bank_tickers[name]
        df = yf.download(ticker, start=start, end=end, interval='1d', progress=False)
        if not df.empty:
            df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df_list.append(df['Returns'].rename(name))
    if not df_list:
        return None
    return pd.concat(df_list, axis=1).dropna()

# -------------------------------
# FUNCTIONS: QAOA
# -------------------------------
def parse_vector(txt):
    vals = [v.strip() for v in txt.split(",") if v.strip() != ""]
    return np.array([float(v) for v in vals], dtype=float)

def parse_matrix(txt, n):
    rows = [r for r in txt.strip().split("\n") if r.strip() != ""]
    M = []
    for r in rows:
        M.append([float(v.strip()) for v in r.split(",") if v.strip() != ""])
    M = np.array(M, dtype=float)
    if M.shape != (n, n):
        raise ValueError(f"Covariance must be {n}x{n}, got {M.shape}")
    return 0.5 * (M + M.T)

def is_pos_semidefinite(M, tol=1e-10):
    eig = np.linalg.eigvalsh(M)
    return np.all(eig > -tol)

def markowitz_to_ising(mu, Sigma, lam, k, P):
    n = len(mu)
    Q_lin = -mu + (P*(1 - 2*k)) * np.ones(n)
    Q_quad = lam * Sigma.copy()
    for i in range(n):
        for j in range(i+1, n):
            Q_quad[i, j] += 2*P
            Q_quad[j, i] += 2*P
    h = np.zeros(n)
    J = np.zeros((n, n))
    const = 0.0
    const += 0.5 * np.sum(Q_lin)
    h += -0.5 * Q_lin
    for i in range(n):
        for j in range(i+1, n):
            qij = Q_quad[i, j]
            const += 0.25 * qij
            h[i] += -0.25 * qij
            h[j] += -0.25 * qij
            J[i, j] += 0.25 * qij
            J[j, i] += 0.25 * qij
    return h, J, const

def exp_ising_layer(qc, h, J, gamma):
    n = len(h)
    for i in range(n):
        angle = 2.0 * gamma * h[i]
        if abs(angle) > 1e-12:
            qc.rz(angle, i)
    for i in range(n):
        for j in range(i+1, n):
            if abs(J[i, j]) > 1e-12:
                qc.cx(i, j)
                qc.rz(2.0 * gamma * J[i, j], j)
                qc.cx(i, j)

def build_qaoa_ising_circuit(h, J, beta, gamma):
    n = len(h)
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.h(i)
    exp_ising_layer(qc, h, J, gamma)
    for i in range(n):
        qc.rx(2.0 * beta, i)
    qc.barrier()
    qc.measure(range(n), range(n))
    return qc

def simulate_counts(qc, shots=2048):
    sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
    probs = sv.probabilities_dict()
    counts = {k[::-1]: int(v * shots) for k, v in probs.items()}
    return counts

def bitstring_to_x(bitstring):
    return np.array([int(b) for b in bitstring], dtype=int)

def portfolio_metrics(mu, Sigma, x):
    ret = float(mu @ x)
    var = float(x @ Sigma @ x)
    return ret, var

def objective_value(mu, Sigma, lam, x):
    ret, var = portfolio_metrics(mu, Sigma, x)
    return ret - lam * var

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    start = dt.date.today() - dt.timedelta(days=30)
    end = dt.date.today()
    selected = ["HDFCBANK", "ICICIBANK"]  # example banks

    df = get_data_selected(selected, start, end)
    if df is None:
        print("No data found.")
        exit()

    mu = df.mean().values
    Sigma = df.cov().values
    print("μ:", mu)
    print("Σ:\n", Sigma)

    # Parameters
    k = 2
    lam = 0.5
    P = 5.0
    beta = 0.7
    gamma = 0.9

    h, J, const = markowitz_to_ising(mu, Sigma, lam, k, P)
    qc = build_qaoa_ising_circuit(h, J, beta, gamma)
    qc.draw("mpl")

    counts = simulate_counts(qc, shots=2048)
    print("\n📊 Simulated counts (top):", dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]))
    plot_histogram(counts, figsize=(6, 4))
    plt.show()

    best = max(counts, key=counts.get)
    x = bitstring_to_x(best)
    if x.sum() != k:
        idx_sorted = np.argsort(x)[::-1]
        x[:] = 0
        x[idx_sorted[:k]] = 1

    exp_ret, exp_var = portfolio_metrics(mu, Sigma, x)
    obj = objective_value(mu, Sigma, lam, x)

    print("\n🏆 Best portfolio:", x.tolist())
    print(f"📈 Return: {exp_ret:.4f}, 📉 Variance: {exp_var:.4f}, 🎯 Objective: {obj:.4f}")
