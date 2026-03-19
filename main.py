import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st
import yaml
from pathlib import Path


def load_config(path: str = "config.yaml") -> dict:
    with open(Path(__file__).parent / path) as f:
        return yaml.safe_load(f)


# odes
def sir_model(t, y, beta, gamma):
    S, I, R = y
    dS = -beta * S * I
    dI =  beta * S * I - gamma * I
    dR =  gamma * I
    return [dS, dI, dR]


def seir_model(t, y, beta, sigma, gamma):
    S, E, I, R = y
    dS = -beta * S * I
    dE =  beta * S * I - sigma * E
    dI =  sigma * E - gamma * I
    dR =  gamma * I
    return [dS, dE, dI, dR]


def seirs_model(t, y, beta, sigma, gamma, xi):
    S, E, I, R = y
    dS = -beta * S * I + xi * R
    dE =  beta * S * I - sigma * E
    dI =  sigma * E - gamma * I
    dR =  gamma * I - xi * R
    return [dS, dE, dI, dR]


def sird_model(t, y, beta, gamma, mu):
    S, I, R, D = y
    dS = -beta * S * I
    dI =  beta * S * I - gamma * I - mu * I
    dR =  gamma * I
    dD =  mu * I
    return [dS, dI, dR, dD]


# functions
def run_sir(R0, infectious_period, N, days, I0):
    gamma = 1 / infectious_period
    beta  = R0 * gamma
    S0    = (N - I0) / N

    sol = solve_ivp(
        sir_model, (0, days), [S0, I0 / N, 0.0],
        t_eval=np.linspace(0, days, days * 4),
        args=(beta, gamma), method="RK45", rtol=1e-6,
    )
    S, I, R = sol.y * 100
    peak_idx = np.argmax(I)
    return {
        "model": "SIR", "t": sol.t,
        "compartments": {"Susceptible": S, "Infectious": I, "Recovered": R},
        "R0": R0, "beta": beta, "gamma": gamma,
        "infectious_period": infectious_period, "N": N,
        "peak_pct": I[peak_idx], "peak_day": sol.t[peak_idx],
        "total_pct": R[-1],
        "herd_threshold": (1 - 1 / R0) * 100 if R0 > 1 else 0.0,
    }


def run_seir(R0, infectious_period, incubation_period, N, days, I0):
    gamma = 1 / infectious_period
    sigma = 1 / incubation_period
    beta  = R0 * gamma
    S0    = (N - I0) / N

    sol = solve_ivp(
        seir_model, (0, days), [S0, 0.0, I0 / N, 0.0],
        t_eval=np.linspace(0, days, days * 4),
        args=(beta, sigma, gamma), method="RK45", rtol=1e-6,
    )
    S, E, I, R = sol.y * 100
    peak_idx = np.argmax(I)
    return {
        "model": "SEIR", "t": sol.t,
        "compartments": {"Susceptible": S, "Exposed": E, "Infectious": I, "Recovered": R},
        "R0": R0, "beta": beta, "sigma": sigma, "gamma": gamma,
        "infectious_period": infectious_period, "incubation_period": incubation_period, "N": N,
        "peak_pct": I[peak_idx], "peak_day": sol.t[peak_idx],
        "total_pct": R[-1],
        "herd_threshold": (1 - 1 / R0) * 100 if R0 > 1 else 0.0,
    }


def run_seirs(R0, infectious_period, incubation_period, immunity_duration, N, days, I0):
    gamma = 1 / infectious_period
    sigma = 1 / incubation_period
    xi    = 1 / immunity_duration
    beta  = R0 * gamma
    S0    = (N - I0) / N

    sol = solve_ivp(
        seirs_model, (0, days), [S0, 0.0, I0 / N, 0.0],
        t_eval=np.linspace(0, days, days * 4),
        args=(beta, sigma, gamma, xi), method="RK45", rtol=1e-6,
    )
    S, E, I, R = sol.y * 100
    peak_idx = np.argmax(I)
    return {
        "model": "SEIRS", "t": sol.t,
        "compartments": {"Susceptible": S, "Exposed": E, "Infectious": I, "Recovered": R},
        "R0": R0, "beta": beta, "sigma": sigma, "gamma": gamma, "xi": xi,
        "infectious_period": infectious_period, "incubation_period": incubation_period,
        "immunity_duration": immunity_duration, "N": N,
        "peak_pct": I[peak_idx], "peak_day": sol.t[peak_idx],
        "total_pct": R[-1],
        "herd_threshold": (1 - 1 / R0) * 100 if R0 > 1 else 0.0,
    }


def run_sird(R0, infectious_period, mortality_rate, N, days, I0):
    gamma = 1 / infectious_period
    beta  = R0 * (gamma + mortality_rate)
    S0    = (N - I0) / N

    sol = solve_ivp(
        sird_model, (0, days), [S0, I0 / N, 0.0, 0.0],
        t_eval=np.linspace(0, days, days * 4),
        args=(beta, gamma, mortality_rate), method="RK45", rtol=1e-6,
    )
    S, I, R, D = sol.y * 100
    peak_idx = np.argmax(I)
    return {
        "model": "SIRD", "t": sol.t,
        "compartments": {"Susceptible": S, "Infectious": I, "Recovered": R, "Dead": D},
        "R0": R0, "beta": beta, "gamma": gamma, "mu": mortality_rate,
        "infectious_period": infectious_period, "mortality_rate": mortality_rate, "N": N,
        "peak_pct": I[peak_idx], "peak_day": sol.t[peak_idx],
        "total_pct": R[-1],
        "herd_threshold": (1 - 1 / R0) * 100 if R0 > 1 else 0.0,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "Susceptible": "#3c8c5c",
    "Exposed":     "#e8a020",
    "Infectious":  "#d85a30",
    "Recovered":   "#378add",
    "Dead":        "#555555",
}


def plot_model(res):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for label, values in res["compartments"].items():
        color = COLORS[label]
        ax.plot(res["t"], values, color=color, lw=2, label=label)
        if label == "Infectious":
            ax.fill_between(res["t"], values, alpha=0.15, color=color)

    ax.axvline(res["peak_day"], color=COLORS["Infectious"], lw=1, ls="--", alpha=0.6)
    ax.axhline(res["peak_pct"], color=COLORS["Infectious"], lw=1, ls="--", alpha=0.6)
    ax.annotate(
        f"Peak {res['peak_pct']:.1f}% on day {res['peak_day']:.0f}",
        xy=(res["peak_day"], res["peak_pct"]),
        xytext=(res["peak_day"] + 10, res["peak_pct"] + 3),
        fontsize=9, color=COLORS["Infectious"],
        arrowprops=dict(arrowstyle="->", color=COLORS["Infectious"], lw=0.8),
    )

    ax.set_xlabel("Days", fontsize=11)
    ax.set_ylabel("% of population", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, res["t"][-1])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(loc="center right", framealpha=0.9)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        f"{res['model']} model  |  R₀ = {res['R0']}  |  1/γ = {res['infectious_period']} days  |  N = {res['N']:,}",
        fontsize=12, pad=10,
    )
    fig.tight_layout()
    return fig


# streamlit

cfg = load_config()
d   = cfg["defaults"]

st.set_page_config(page_title="Measles Epidemic Model", layout="wide")
st.title("Measles Epidemic Model")

with st.sidebar:
    st.header("Model")
    model = st.selectbox("Model type", ["SIR", "SEIR", "SEIRS", "SIRD"])

    st.header("Parameters")
    R0                = st.slider("Basic reproduction number (R₀)", 1.0, 18.0, float(d["R0"]), 0.1)
    infectious_period = st.slider("Infectious period (days)", 1, 30, int(d["infectious_period"]))
    N                 = st.number_input("Population size (N)", min_value=100, max_value=10_000_000, value=int(d["N"]), step=1000)
    days              = st.slider("Simulation duration (days)", 30, 730, int(d["days"]))
    I0                = st.number_input("Initial infectious (I₀)", min_value=1, max_value=1000, value=int(d["I0"]))

    if model in ("SEIR", "SEIRS"):
        incubation_period = st.slider("Incubation period (days)", 1, 21, int(d["incubation_period"]))
    if model == "SEIRS":
        immunity_duration = st.slider("Immunity duration (days)", 30, 3650, int(d["immunity_duration"]))
    if model == "SIRD":
        mortality_rate = st.slider("Daily mortality rate (μ)", 0.0, 0.1, float(d["mortality_rate"]), 0.001, format="%.3f")

# Run selected model
if model == "SIR":
    res = run_sir(R0, infectious_period, int(N), days, int(I0))
elif model == "SEIR":
    res = run_seir(R0, infectious_period, incubation_period, int(N), days, int(I0))
elif model == "SEIRS":
    res = run_seirs(R0, infectious_period, incubation_period, immunity_duration, int(N), days, int(I0))
else:
    res = run_sird(R0, infectious_period, mortality_rate, int(N), days, int(I0))

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("R₀", f"{res['R0']:.2f}")
col2.metric("Peak infectious", f"{res['peak_pct']:.1f}%", f"day {res['peak_day']:.0f}")
col3.metric("Total recovered", f"{res['total_pct']:.1f}%")
col4.metric("Herd immunity threshold", f"{res['herd_threshold']:.0f}%")

st.pyplot(plot_model(res))

with st.expander("Derived parameters"):
    st.write(f"- β (transmission rate): `{res['beta']:.4f}` /day")
    st.write(f"- γ (recovery rate): `{res['gamma']:.4f}` /day")
    if "sigma" in res:
        st.write(f"- σ (incubation rate): `{res['sigma']:.4f}` /day")
    if "xi" in res:
        st.write(f"- ξ (waning immunity rate): `{res['xi']:.4f}` /day")
    if "mu" in res:
        st.write(f"- μ (mortality rate): `{res['mu']:.4f}` /day")
