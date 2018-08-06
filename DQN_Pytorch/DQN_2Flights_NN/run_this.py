from maze_env import Maze
from RL_brain import DQN

def run_maze():
    step = 0
    achieved_step_min = 1000
    check_success_episode = 0
    success_rate = 0
    for episode in range(500000):
        # initial observation
        observation, suggest_action_num = env.reset()
        action_record = ''
        action_step = 0
        force_suggest = False
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation, suggest_action_num, force_suggest)

            if episode < 500:
                ignore_crash = True
            else:
                if episode == 500:
                    print('stop ignore crash')
                ignore_crash = False
            # RL take action and get next observation and reward
            observation_, reward, done, achieved, suggest_action_num, can_be_stored, force_suggest = env.step(action, ignore_crash)

            if can_be_stored:
                RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
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
