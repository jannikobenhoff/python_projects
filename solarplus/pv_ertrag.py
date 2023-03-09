import math
import numpy as np

pvWg = 0.215

# Silizium
si_koeff = {
  "U0": 26.9,
  "U1": 6.20,
  "k1": -0.017237,
  "k2": -0.040465,
  "k3": -0.004702,
  "k4": 0.000149,
  "k5": 0.000170,
  "k6": 0.000005,
}

# Cadmiumtellurid
cdte_koeff = {
  "U0": 23.4,
  "U1": 5.44,
  "k1": -0.046689,
  "k2": -0.072844,
  "k3": -0.002262,
  "k4": 0.000276,
  "k5": 0.000159,
  "k6": -0.000006
}


def pv_performance(pv, Gin, Temp, W_mod):
    if pv == "Si":
        koeff = si_koeff
    else:
        koeff = cdte_koeff
    print(pv)
    T_mod = Temp + Gin / (koeff["U0"] + koeff["U1"] * W_mod)
    T = T_mod - 25
    G = Gin / 1000
    P_stc = 330
    P = G * P_stc * (1 +
          koeff["k1"] * np.log(G) +
          koeff["k2"] * pow(np.log(G), 2) +
          koeff["k3"] * T +
          koeff["k4"] * T * np.log(G) +
          koeff["k5"] * T * pow(np.log(G), 2) +
          koeff["k6"] * pow(T, 2))
    return P * pvWg

