from SlimeSimSettings import Settings
import random
import math
import numpy as np


class SlimeSimNormal:
    def __init__(self):
        self.settings = Settings()
        self.size = self.settings.size - 2
        self.field = np.zeros(((self.size + 2) ** 2), dtype=float)
        self.agents = []
        for i in range(self.settings.amountOfAgents):
            xpos = (random.uniform(0., 1.)) * self.size
            ypos = (random.uniform(0., 1.)) * self.size
            angle = random.uniform(0., 1.) * 2 * math.pi
            self.agents += [[xpos, ypos, angle]]

    def advance(self, timestep: float):
        self.advanceAgents(timestep)
        self.blurDiffuse(timestep)

    def step(self):
        self.advance(self.settings.dt)

    def advanceAgents(self, timestep: float):
        for agent in self.agents:
            self.slimesteerUpdate(agent, timestep)

            dirx = math.cos(agent[2])
            newx = agent[0] + dirx * self.settings.speed * timestep

            if newx < 0.0 or newx > self.size:
                newx = np.clip(newx, 0.0, (self.size - 1) * 1.0)
                agent[2] = -random.uniform(0., 1.) * 2 * math.pi

            diry = math.sin(agent[2])
            newy = agent[1] + diry * self.settings.speed * timestep

            if newy < 0.0 or newy > self.size:
                newy = np.clip(newy, 0.0, (self.size - 1) * 1.0)
                agent[2] = -random.uniform(0., 1.) * 2 * math.pi

            agent[0] = newx
            agent[1] = newy

            self.field[(int(newx) + 1) + (int(newy) + 1) * self.size] = 1.0

    def slimesteerUpdate(self, agent: list, timestep: float):
        wForward = self.sense(agent, 0)
        wLeft = self.sense(agent, math.pi / 4)
        wRight = self.sense(agent, -(math.pi / 4))

        if wForward > wLeft and wForward > wRight:
            return
        if wLeft > wRight:
            agent[2] += self.settings.steerStrength * timestep
        else:
            agent[2] -= self.settings.steerStrength * timestep

    def sense(self, agent: list, angle: float):
        sangle = agent[2] + angle
        sdirx = math.cos(sangle)
        sposx = agent[0] + sdirx + self.settings.sensorDirOffset
        sdiry = math.sin(sangle)
        sposy = agent[1] + sdiry + self.settings.sensorDirOffset

        sum = 0.
        for i in range(-self.settings.sensorSize, self.settings.sensorSize):
            for j in range(-self.settings.sensorSize, self.settings.sensorSize):
                cx = int(sposx + i)
                cy = int(sposy + j)
                if 0 < cx < self.size and 0 < cy < self.size:
                    sum += self.field[(cx + 1) + (cy + 1) * self.size]
        return sum

    def interpolate(self, a: float, b: float, t: float):
        return a + t * (b - a)

    def blurDiffuse(self, timestep: float):
        for i in range(self.size):
            for j in range(self.size):
                self.field[(i + 1) + (j + 1) * self.size] = max(0., self.field[(i + 1) + (j + 1) * self.size] - (
                        self.settings.evaporate * timestep))
        for i in range(self.size):
            for j in range(self.size):
                sum = 0.
                for dx in range(-1,1):
                    for dy in range(-1,1):
                        sx = i + dx
                        sy = j + dy
                        sum += self.field[(sx + 1) + (sy + 1) * self.size]
                self.field[(i + 1) + (j + 1) * self.size] = self.interpolate(self.field[(i + 1) + (j + 1) * self.size],
                                                                             sum / 9.0,
                                                                             10 * timestep)

    def getf(self):
        return self.field.reshape((self.size + 2, self.size + 2))

    def step_and_render_frame(self, frame_num=None, img=None):
        self.step()
        if img:
            img.set_array(self.field.reshape((self.size + 2, self.size + 2)))
        return (img,)
