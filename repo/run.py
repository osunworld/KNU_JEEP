import time
from autocar3g.driving import Driving


class TrackController:
    def __init__(self, direction="right"):
        assert direction in ["right", "left"]

        self.direction = direction
        self.DIR = 1 if direction == "right" else -1

        self.car = Driving()

        self.THROTTLE_STRAIGHT = 9
        self.THROTTLE_TURN_1  = 4
        self.THROTTLE_TURN_2  = 6

        self.STEER_CENTER = 0.0

        self.T_STRAIGHT_1 = 1.32
        self.T_STRAIGHT_2 = 0.55

    def straight(self, duration):
        self.car.steering = self.STEER_CENTER
        time.sleep(0.1)

        self.car.throttle = self.THROTTLE_STRAIGHT
        time.sleep(duration)

        self.car.throttle = 0
        time.sleep(0.25)

    def turn_entry(self):
        for steer in [0.1, 0.17, 0.25, 0.35]:
            self.car.steering = self.DIR * steer
            time.sleep(0.08)

        self.car.throttle = self.THROTTLE_TURN_1
        time.sleep(2.02)

        self.car.steering = self.DIR * 0.45
        time.sleep(0.25)

        self.car.steering = self.DIR * 0.08
        time.sleep(0.1)

    def turn_connect(self):
        for steer in [0.25, 0.50, 0.52]:
            self.car.steering = self.DIR * steer
            time.sleep(0.07)

        time.sleep(0.65)

        self.car.throttle = self.THROTTLE_TURN_2
        time.sleep(1.92)

        for steer in [0.35, 0.20, 0.0]:
            self.car.steering = self.DIR * steer
            time.sleep(0.08)

        self.car.throttle = 0
        self.car.steering = self.STEER_CENTER
        time.sleep(0.4)

    def run(self):
        try:
            print(f"▶ {self.direction.upper()} TRACK START")

            self.straight(self.T_STRAIGHT_1)
            self.turn_entry()
            self.turn_connect()

            self.straight(self.T_STRAIGHT_2)
            self.turn_entry()
            self.turn_connect()

            print("✅ TRACK COMPLETE")

        finally:
            self.stop()

    def stop(self):
        print("🛑 EMERGENCY STOP SEQUENCE")

        # throttle / steering 즉시 해제
        for _ in range(5):
            self.car.throttle = 0
            self.car.steering = 0
            time.sleep(0.1)

        # 관성 제거용 미세 역브레이크
        self.car.throttle = 0
        self.car.steering = 0
        time.sleep(0.15)

        # 완전 정지 latch 유지
        for _ in range(10):
            self.car.throttle = 0
            self.car.steering = 0
            time.sleep(0.1)

        print("✅ VEHICLE FULLY STOPPED")


if __name__ == "__main__":
    import random
    
    r = random.randint(1,100)
    if r <= 50:
        direction = 'right'
    else:
        direction = 'left'
    controller = TrackController(direction=direction)
    controller.run()
