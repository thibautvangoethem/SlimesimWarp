from SlimeSimSettings import Settings
import random
from numba import jit
import numpy as np


@jit(nopython=True, nogil=True)
def interpolate(a, b, t):
    return a + t * (b - a)

@jit(nopython=True, nogil=True)
def advanceAgents(timestep: float, agents, size, field, speed, steerStrength, sensorDirOffset, sensorSize,
                  amountOfAgents):
    for agentIter in range(amountOfAgents):
        wForward = sense(agents[agentIter], 0, sensorDirOffset, sensorSize, size, field)
        wLeft = sense(agents[agentIter], np.pi / 4, sensorDirOffset, sensorSize, size, field)
        wRight = sense(agents[agentIter], -(np.pi / 4), sensorDirOffset, sensorSize, size, field)

        if not (wForward > wLeft and wForward > wRight):
            if wLeft > wRight:
                agents[agentIter][2] += steerStrength * timestep
            else:
                agents[agentIter][2] -= steerStrength * timestep

        dirx = np.cos(agents[agentIter][2])
        newx = agents[agentIter][0] + dirx * speed * timestep

        if newx < 0.0 or newx > size:
            newx = min(max(newx, 0.0), (size - 1))
            agents[agentIter][2] = -random.uniform(0., 1.) * 2 * np.pi

        diry = np.sin(agents[agentIter][2])
        newy = agents[agentIter][1] + diry * speed * timestep

        if newy < 0.0 or newy > size:
            newy = min(max(newy, 0.0), (size - 1))
            agents[agentIter][2] = -random.uniform(0., 1.) * 2 * np.pi

        agents[agentIter][0] = newx
        agents[agentIter][1] = newy

        field[(int(newx)) + (int(newy)) * size] = 1.0

@jit(nopython=True, nogil=True)
def slimesteerUpdate(agent, timestep: float, steerStrength, sensorDirOffset, sensorSize, size, field):
    wForward = sense(agent, 0, sensorDirOffset, sensorSize, size, field)
    wLeft = sense(agent, np.pi / 4, sensorDirOffset, sensorSize, size, field)
    wRight = sense(agent, -(np.pi / 4), sensorDirOffset, sensorSize, size, field)

    if wForward > wLeft and wForward > wRight:
        return
    if wLeft > wRight:
        agent[2] += steerStrength * timestep
    else:
        agent[2] -= steerStrength * timestep

@jit(nopython=True, nogil=True)
def sense(agent: list, angle: float, sensorDirOffset, sensorSize, size, field):
    sangle = agent[2] + angle
    sdirx = np.cos(sangle)
    sposx = agent[0] + sdirx * sensorDirOffset
    sdiry = np.sin(sangle)
    sposy = agent[1] + sdiry * sensorDirOffset

    sum = 0.
    for i in range(-sensorSize, sensorSize):
        cx = int(sposx + i)
        if cx >= 0 and cx < size:
            continue
        for j in range(-sensorSize, sensorSize):
            cy = int(sposy + j)
            if cy >= 0 and cy < size:
                sum += field[(cx) + (cy) * size]
    return sum

@jit(nopython=True, nogil=True)
def blurDiffuse(timestep, size, field, evaporate):
    for i in range(size):
        for j in range(size):
            field[(i) + (j) * size] = max(0., field[(i) + (j) * size] - (
                    evaporate * timestep))
    for i in range(size):
        for j in range(size):
            sum = 0.
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    sx = i + dx
                    sy = j + dy
                    if (sx >= 0 and sx < size and sy >= 0 and sy < size):
                        sum += field[(sx) + (sy) * size]

            field[(i) + (j) * size] = interpolate(field[(i) + (j) * size],
                                                          sum / 9.0,
                                                          10 * timestep)

class SlimeSimNumba(object):
    def __init__(self):
        settings = Settings()
        self.speed = settings.speed
        self.steerStrength = settings.steerStrength
        self.sensorDirOffset = settings.sensorDirOffset
        self.sensorSize = settings.sensorSize
        self.amountOfAgents = settings.amountOfAgents
        self.evaporate = settings.evaporate
        self.size = settings.size
        self.dt=settings.dt

        self.field = np.zeros(((self.size) ** 2), dtype=float)
        self.agents = np.zeros((settings.amountOfAgents, 3))
        for i in range(settings.amountOfAgents):
            self.agents[i][0] = (random.uniform(0., 1.)) * self.size
            self.agents[i][1] = (random.uniform(0., 1.)) * self.size
            self.agents[i][2] = random.uniform(0., 1.) * 2 * np.pi

    def advance(self, timestep: float):
        advanceAgents(timestep, self.agents, self.size, self.field, self.speed, self.steerStrength,
                      self.sensorDirOffset, self.sensorSize, self.amountOfAgents)
        blurDiffuse(timestep, self.size, self.field, self.evaporate)

    def step(self):
        self.advance(self.dt)

    def step_and_render_frame(self, frame_num=None, img=None):
        self.step()
        if img:
            img.set_array(self.field.reshape((self.size, self.size)))

        return (img,)

    def getf(self):
        return self.field.reshape((self.size, self.size))

if __name__ == '__main__':
    s = Settings()
    sim = SlimeSimNumba(512, s)
    while True:
        sim.advance(0.05)
    # interpolate.parallel_diagnostics(level=4)
