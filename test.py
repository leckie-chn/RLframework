

from DPG import DPGAgent

import matplotlib.pyplot as plt


# tensorflow

agent = DPGAgent(max_round=200, n_sample=20, batch_size=10, gamma=0.0)

# TODO: profiling for A3C algorithm
agent.train()
plt.figure()
plt.plot(agent.loss_history, 'g-')
plt.show()
plt.close()

