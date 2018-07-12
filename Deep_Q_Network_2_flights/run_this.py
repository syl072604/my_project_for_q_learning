from maze_env import Maze
from RL_brain import DeepQNetwork
import numpy as np


def run_maze():
    step = 0
    achieved_step_min = 1000
    for episode in range(1000):
        # initial observation
        observation = env.reset()
        action_record = ''
        action_step = 0
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, achieved = env.step(action)

            if (observation_ == observation).all():
                continue

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            observation = observation_

            action_step += 1

            action_record = action_record + str(env.action_space[action]) + '\n'

            if achieved:
                if action_step<achieved_step_min:
                    file = open('action_record.txt','w')
                    achieved_step_min = action_step
                    file.write(str(achieved_step_min) + '\n' + action_record )
                    file.close()

            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features, env.n_flights,
                      action_space=env.action_space,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()