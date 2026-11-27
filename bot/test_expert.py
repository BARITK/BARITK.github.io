import numpy as np, time, sys, train_padel_bot as T
from expert import expert_pointer
def run(n=64, cap=2000, seed=1, **kw):
    rng=np.random.default_rng(seed); env=T.Env(n,rng)
    for s in range(100000):
        px,py=expert_pointer(env,**kw); env.step_pointer(px,py)
        env.done |= env.score>=cap
        if env.done.all(): break
    return env.score
t=time.time(); sc=run()
print('expert: mean %.0f median %.0f min %.0f  >=1500: %d/64  (%.0fs)'%(sc.mean(),np.median(sc),sc.min(),(sc>=1500).sum(),time.time()-t))
