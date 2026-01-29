class PIDControllerV2:
    def __init__(
        self,
        kp=0.8,
        kd=0.04,
        steer_gain=4.5,
        max_steer=1.0,
        deadzone=0.03,
        corner_boost=1.3
    ):
        self.kp = kp
        self.kd = kd
        self.steer_gain = steer_gain
        self.max_steer = max_steer
        self.deadzone = deadzone
        self.corner_boost = corner_boost
        self.prev_error = 0.0

    def compute(self, error):
        if abs(error) < self.deadzone:
            error = 0.0

        d_error = error - self.prev_error
        self.prev_error = error

        raw = self.kp * error + self.kd * d_error

        gain = self.steer_gain
        if abs(error) > 0.20:
            gain *= self.corner_boost

        steering = gain * raw

        if steering > self.max_steer:
            steering = self.max_steer
        elif steering < -self.max_steer:
            steering = -self.max_steer

        return steering
