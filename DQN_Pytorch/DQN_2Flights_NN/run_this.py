from maze_env import Maze
from RL_brain import DQN
import matplotlib.pyplot as plt
import numpy as np
def run_maze():
    step = 0
    achieved_step_min = 1000
    check_success_episode = 0
    success_rate = 0
    state_episode = 0
    change_origin = True
    rate_reset = False
    success_rate_plot = []
    for episode in range(500000):
        # initial observation
        observation, suggest_action_num, distance_too_large = env.reset(change_origin)
        if distance_too_large:
            env.change_input()
            state_episode = 0
            RL.change_input_count()
            print('distance too large change input:', RL.change_input)
        action_record = ''
        action_step = 0
        force_suggest = False
        state_episode += 1
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation, suggest_action_num, force_suggest)

            if state_episode < 20:
                change_origin = True
                ignore_crash = True
            elif state_episode < 50:
                change_origin = False
                ignore_crash = True
            else:
                if state_episode == 50:
                    print('stop ignore crash for this input')
                ignore_crash = False
            # RL take action and get next observation and reward
            observation_, reward, done, achieved, suggest_action_num, can_be_stored, force_suggest = env.step(action, ignore_crash, change_origin)

            if can_be_stored:
                RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (not ignore_crash) and (step % 5 == 0):
                RL.learn()

            observation = observation_

            action_step += 1

            action_record = action_record + str(env.action_space[action]) + '\n'

            if achieved:
                if action_step < achieved_step_min:
                    file = open('action_record.txt', 'w')
                    achieved_step_min = action_step
                    file.write(str(achieved_step_min) + '\n' + action_record )
                    file.close()

            if done:
                if not ignore_crash:
                    if check_success_episode < 100:
                        check_success_episode += 1
                        if achieved:
                            success_rate += 1
                    else:
                        print('success rate: %d percent' % success_rate)
                        if rate_reset:
                            plt.close()
                            success_rate_plot.append(success_rate)
                            plt.plot(np.array(success_rate_plot), c='r', label='success_rate_plot')
                            plt.show()
                        if success_rate > 80 or state_episode > 5000:
                            rate_reset = True
                            env.change_input()
                            state_episode = 0
                            RL.change_input_count()
                            print('change input:', RL.change_input)
                        else:
                            rate_reset = False
                        check_success_episode = 0
                        success_rate = 0

                break

            if action_step > 50:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DQN(n_actions=env.n_actions, n_features=env.n_features, n_flights=env.n_flights,action_space=env.action_space)
    env.after(100, run_maze)
    env.mainloop()
