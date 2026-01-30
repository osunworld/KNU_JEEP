class PIDControllerV4:
    def __init__(self, kp=0.9, ki=0.02, kd=0.05, deadzone=0.03):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.deadzone = deadzone

        self.prev_error = 0.0
        self.i_error = 0.0

    def compute(self, error):
        if abs(error) < self.deadzone:
            error = 0.0

        self.i_error += error
        self.i_error = max(-1.0, min(1.0, self.i_error))

        d_error = error - self.prev_error
        self.prev_error = error

        return (
            self.kp * error
            + self.ki * self.i_error
            + self.kd * d_error
        )
