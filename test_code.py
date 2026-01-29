import numpy as np
from agent import BanditAgent

# 미리 생성된 보상 데이터 (1000 steps x 2 arms)
rewards_data = np.load("rwd_seq_example_01.npy")

def evaluate(agent):
    total_reward = 0
    for t in range(len(rewards_data)):
        chosen_arm = agent.select_arm()
        reward = rewards_data[t, chosen_arm] # 실제 환경 대신 시퀀스에서 추출
        agent.update(chosen_arm, reward)
        total_reward += reward
    return total_reward


output = evaluate(BanditAgent())
print("Total # of rewards :", output)