# pid_controller.py

class PIDController:
    def __init__(self, kp=0.8, kd=0.15, max_steer=0.7):
        self.kp = kp
        self.kd = kd
        self.max_steer = max_steer
        self.prev_error = 0.0

    def compute(self, error):
        d_error = error - self.prev_error
        self.prev_error = error

        steering = self.kp * error + self.kd * d_error
        steering = max(-self.max_steer, min(self.max_steer, steering))
        return steering
