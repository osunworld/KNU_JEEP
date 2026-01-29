# 아래 코드 작성,

class BanditAgent:
  def __init__(self, n_arms=2):
    self.n_arms = n_arms
    # 필요한 변수 초기화

  def select_arm(self):
    # L(0)/R(1) 선택 로직 (0 또는 1 반환)
    return 0

  def update(self, arm, reward):
    # 결과 학습 로직
    pass
