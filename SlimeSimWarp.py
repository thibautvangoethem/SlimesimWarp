import math
import warp as wp
import random
import numpy as np
from SlimeSimSettings import Settings

s = Settings()
si = s.size
size = wp.constant(si)
# used for some calculations that require float size, dont want to cast each time
sizef = wp.constant(float(si))

amountOfAgents = wp.constant(s.amountOfAgents)
speed = wp.constant(s.speed)
evaporate = wp.constant(s.evaporate)
steerStrength = wp.constant(s.steerStrength)
sensorSize = wp.constant(s.sensorSize)
sensorDirOffset = wp.constant(s.sensorDirOffset)
diffuseSize = wp.constant(s.diffuseSize)
diffuseSizef = wp.constant(float(((s.diffuseSize * 2) + 1) ** 2))
randomSteerForce = wp.constant(s.randomSteerForce)
dt = wp.constant(s.dt)


# wp.config.print_launches = True
# wp.config.mode = "debug"


@wp.func
def sense(agent: wp.vec3f, field: wp.array2d(dtype=float), angle: float):
    sangle = agent[2] + angle
    sdirx = wp.cos(sangle)
    sposx = int(agent[0] + sdirx * sensorDirOffset)
    sdiry = wp.sin(sangle)
    sposy = int(agent[1] + sdiry * sensorDirOffset)

    sum = float(0.)
    for i in range(-sensorSize, sensorSize):
        for j in range(-sensorSize, sensorSize):
            cx = sposx + i
            cy = sposy + j
            if 0 <= cx < size and 0 <= cy < size:
                sum += field[cx, cy]
    return sum


@wp.func
def slimesteerUpdate(agent: wp.vec3f, field: wp.array2d(dtype=float), seed: int,tid: int):
    wForward = sense(agent, field, 0.)
    wLeft = sense(agent, field, wp.pi / 4.)
    wRight = sense(agent, field, -(wp.pi / 4.))
    rng = wp.rand_init(seed,tid)
    randomsteer = wp.randf(rng, -1.0, 1.0) * randomSteerForce

    if wForward > wLeft and wForward > wRight:
        return agent[2] + randomsteer
    if wLeft > wRight:
        return agent[2] + (steerStrength + randomsteer) * dt
    else:
        return agent[2] + (-steerStrength + randomsteer) * dt


@wp.kernel
def advanceAgents(agents: wp.array(dtype=wp.vec3f), field: wp.array2d(dtype=float), seed: int):
    i = wp.tid()
    rng = wp.rand_init(seed, wp.tid())

    agent = agents[i]

    newangle = slimesteerUpdate(agent, field, seed,i)

    dirx = wp.cos(newangle)
    newx = agent[0] + dirx * speed * dt

    if newx < 0.0 or newx > size:
        newx=wp.randf(rng,0.,sizef)
        # newx = wp.clamp(newx, 0.0, (sizef - 1.0) * 1.0)
        # newangle = wp.randf(rng) * 2. * math.pi

    diry = wp.sin(newangle)
    newy = agent[1] + diry * speed * dt

    if newy < 0.0 or newy > size:
        newy = wp.randf(rng, 0., sizef)
        # newy = wp.clamp(newy, 0.0, (sizef - 1.) * 1.0)
        # newangle = wp.randf(rng) * 2. * math.pi

    agent[0] = newx
    agent[1] = newy
    agent[2] = newangle
    field[int(newx), int(newy)] = 1.0
    agents[i] = agent


@wp.func
def lookup_float(f: wp.array2d(dtype=float), x: int, y: int):
    x = wp.clamp(x, 0, size)
    y = wp.clamp(y, 0, size)

    return f[x, y]


@wp.kernel
def blurdiffuse(field: wp.array2d(dtype=float)):
    i, j = wp.tid()

    field[i, j] = wp.max(0., field[i, j] - (evaporate * dt))

    sum = float(0.)
    for dx in range(-diffuseSize, diffuseSize):
        for dy in range(-diffuseSize, diffuseSize):
            sx = i + dx
            sy = j + dy
            if 0 <= sx < size and 0 <= sy < size:
                sum += field[sx, sy]

    field[i, j] = wp.lerp(field[i, j], sum / diffuseSizef, 10.0 * dt)


class SlimeSimWarp:
    def __init__(self):
        shape = (size, size)
        self.field = wp.zeros(shape, dtype=float)
        self.iterations = 100
        self.seed = 0
        npagents = np.zeros((amountOfAgents, 3), dtype=float)
        for i in range(amountOfAgents):
            npagents[i][0] = (random.uniform(0., 1.)) * size
            npagents[i][1] = (random.uniform(0., 1.)) * size
            npagents[i][2] = random.uniform(0., 1.) * 2 * wp.pi
        self.agents = wp.array(npagents, dtype=wp.vec3f)

        # I dont understand this, but the warp example used this and it makes things faster by like 2x, neat
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.advanceAgents()
                wp.launch(blurdiffuse, dim=self.field.shape, inputs=[self.field])
            self.graph = capture.graph

    def getf(self):
        return self.field.numpy()

    def advanceAgents(self):
        wp.launch(advanceAgents, dim=self.agents.size, inputs=[self.agents, self.field, self.seed])

    def step(self):
        with wp.ScopedTimer("step", synchronize=True):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.advanceAgents()
                wp.launch(blurdiffuse, dim=self.field.shape, inputs=[self.field])
        self.seed += 1

    def step_and_render_frame(self, frame_num=None, img=None):
        self.step()
        if img:
            img.set_array(self.field.numpy())

        return (img,)


if __name__ == "__main__":
    sim = SlimeSimWarp()
    sim.step()
    sim.step()
    sim.step()
    sim.step()
