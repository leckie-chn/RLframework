

from DPG import DPGAgent
import cProfile
import pickle
import sys

# tensorflow

if len(sys.argv) < 2:
    print "usage: python test.py [history file] [profile file](optional)"

agent = DPGAgent(max_round=1000, n_sample=50, batch_size=32, gamma=0.0)

# TODO: profiling for A3C algorithm
if len(sys.argv) == 2:
    cProfile.run(agent.train(), sys.argv[2])
else:
    agent.train()

fl = open(sys.argv[1], 'w')
pickle.dump(agent.history, fl)
fl.close()

