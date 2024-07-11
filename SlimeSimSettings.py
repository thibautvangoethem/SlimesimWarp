class Settings:
    def __init__(self):
        self.size = 2 ** 10
        self.amountOfAgents = 2 ** 16
        self.speed = 100.0
        self.evaporate = 0.50
        self.steerStrength = 2.0
        self.sensorSize = 8
        self.sensorDirOffset = 8.0
        self.dt = 0.01

        #from this point warp implementation only

        # note diffuSize is the size of the check both ways + 1 for the current state itself, so it will check in a square of (size+1)*(size+1) around itself
        # extra note, keep this low enough or the entire thing becomes a smear
        self.diffuseSize = 2
        #increases the force of the random steering value added, honestly takes up quite some compute and is probably not worth it
        self.randomSteerForce=0.0


