"""Naitzat Fig. 3(e) hollow torus: published beta = (1, 2, 1)."""
import numpy as np, blme.tasks.topology.betti_curve as bc
rng = np.random.default_rng(0)
for N in (1000, 2000):
    u = rng.uniform(0, 2*np.pi, N); v = rng.uniform(0, 2*np.pi, N); R, r = 3.0, 1.0
    T = np.c_[(R+r*np.cos(v))*np.cos(u), (R+r*np.cos(v))*np.sin(u), r*np.sin(v)]
    b0, b1 = bc._count_betti(T, maxdim=1)
    print(f"RESULT torus N={N}: beta=({b0}, {b1})   published beta_1 = 2", flush=True)
th = np.linspace(0, 2*np.pi, 400, endpoint=False)
circ = np.c_[np.cos(th), np.sin(th)] + 0.03*rng.standard_normal((400, 2))
f8 = np.r_[np.c_[np.cos(th)-1, np.sin(th)], np.c_[np.cos(th)+1, np.sin(th)]] + 0.03*rng.standard_normal((800, 2))
print(f"RESULT circle: {bc._count_betti(circ)}   figure-8: {bc._count_betti(f8)}", flush=True)
