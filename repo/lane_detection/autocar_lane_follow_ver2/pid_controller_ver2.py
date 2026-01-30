class PIDControllerV2:
    def __init__(self, kp=0.8, kd=0.04, deadzone=0.03):
        self.kp = kp
        self.kd = kd
        self.deadzone = deadzone
        self.prev_error = 0.0

    def compute(self, error):
        if abs(error) < self.deadzone:
            error = 0.0

        d_error = error - self.prev_error
        self.prev_error = error

        return self.kp * error + self.kd * d_error
