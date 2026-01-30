import threading
import sys
import termios
import tty
import time


class HumanOverrideV3:
    def __init__(self, step=0.05, max_bias=0.4, decay=0.96):
        self.bias = 0.0
        self.step = step
        self.max_bias = max_bias
        self.decay = decay
        self.running = True

        self.thread = threading.Thread(target=self._key_loop, daemon=True)
        self.thread.start()

    def _key_loop(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        try:
            while self.running:
                ch = sys.stdin.read(1)
                if ch == "a":
                    self.bias -= self.step
                elif ch == "d":
                    self.bias += self.step
                elif ch == "s":
                    self.bias = 0.0

                if self.bias > self.max_bias:
                    self.bias = self.max_bias
                elif self.bias < -self.max_bias:
                    self.bias = -self.max_bias

                time.sleep(0.01)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def get(self):
        # automatic decay every control loop call
        self.bias *= self.decay
        if abs(self.bias) < 1e-4:
            self.bias = 0.0
        return self.bias

    def stop(self):
        self.running = False
