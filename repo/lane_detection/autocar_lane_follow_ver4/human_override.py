import threading
import sys
import termios
import tty
import time


class HumanOverride:
    def __init__(self, step=0.06, max_bias=0.6):
        self.bias = 0.0
        self.step = step
        self.max_bias = max_bias
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

                self.bias = max(-self.max_bias, min(self.max_bias, self.bias))
                time.sleep(0.02)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def get(self):
        return self.bias

    def stop(self):
        self.running = False
