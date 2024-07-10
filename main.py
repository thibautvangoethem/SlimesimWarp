
import numpy as np
import cProfile, pstats
import time
from SlimeSimNormal import SlimeSimNormal
from SlimeSimNumba import SlimeSimNumba
from SlimeSimSettings import Settings
from SlimeSimWarp import SlimeSimWarp
import matplotlib
import matplotlib.animation as anim
import matplotlib.pyplot as plt

s=Settings()

#sim = SlimeSimNumba()
#sim=SlimeSimNormal()
sim=SlimeSimWarp()

if __name__ == '__main__':
    headless=False
    if headless:
        while True:
            sim.step()
    else:
        sim.step() # run once for numba compile

        fig = plt.figure()

        img = plt.imshow(
            sim.getf(),
            origin="lower",
            animated=True,
            interpolation="antialiased",
        )
        img.set_norm(matplotlib.colors.Normalize(0.0, 1.0))
        seq = anim.FuncAnimation(
            fig,
            sim.step_and_render_frame,
            fargs=(img,),
            frames=100000,
            blit=True,
            interval=8,
            repeat=False,
        )

        plt.show()
