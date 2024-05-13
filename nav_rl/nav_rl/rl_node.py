import numpy as np
import pandas as pd
from sklearn import preprocessing

from agent import Agent
from utils import plot_learning_curve, manage_memory


def reward_function(state, state_, r):
    reward = -10*np.linalg.norm(state['achieved_goal'][0:3] -
                                state['desired_goal'][0:3]) -\
          np.linalg.norm(state['observation'][0:3] -
                         state['achieved_goal'][0:3])
    reward_ = -10*np.linalg.norm(state_['achieved_goal'][0:3] -
                                 state_['desired_goal'][0:3]) -\
          np.linalg.norm(state_['observation'][0:3] -
                         state_['achieved_goal'][0:3])
    return r + 0.1*(reward_ - reward)


if __name__ == '__main__':
    env = gym.make('PandaPush-v3', render=False)
    manage_memory()

    observation, info = env.reset()
    obs_dim = np.shape(observation['observation'])[
        0] + np.shape(observation['achieved_goal'])[0]

    normalizer = preprocessing.Normalizer().fit(
        np.array(pd.read_csv('state_memory.csv')[0:100000]))

    agent = Agent(alpha=0.00005, beta=0.005, warmup=100000,
                  input_dims=(obs_dim,),
                  n_actions=env.action_space.shape[0],
                  h_actions=env.action_space.high[0],
                  l_actions=env.action_space.low[0], tau=0.02,
                  batch_size=512, layer1_size=512, layer2_size=1024, layer3_size=512)
    
    n_games = 40000
    filename = 'plots/' + 'Panda_Push_' + str(n_games) + '_games.png'

    score_history = []
    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        observation, info = env.reset()
        env.render()  # wait the right amount of time to make the rendering real-time

        obs_goal = np.concatenate(
            (observation['observation'], observation['desired_goal']))
        obs_goal = normalizer.transform([obs_goal])[0]
        done = bool(info['is_success'])
        score = 0

        while not done:
            # action1 = env.action_space.sample()  # random action
            current_position_ee = observation['observation'][0:3]
            desired_position_ee = observation['achieved_goal'][0:3]
            suggest_action = desired_position_ee - current_position_ee
            action = agent.choose_action(
                obs_goal, suggest_action=suggest_action)

            observation_, reward, terminated, truncated, info = env.step(
                action.numpy())
            reward = reward_function(observation_, observation, reward)
            env.render()  # wait the right amount of time to make the rendering real-time

            obs_goal_ = np.concatenate(
                (observation_['observation'], observation_['desired_goal']))
            obs_goal_ = normalizer.transform([obs_goal_])[0]
            done = terminated or truncated or bool(info['is_success'])
            agent.remember(obs_goal, action, reward, obs_goal_, done)
            # HER
            if np.linalg.norm(observation_['achieved_goal'][0:3] -
                              observation['achieved_goal'][0:3]) > 0.002 and observation_['achieved_goal'][2] > 0.01:

                her_reward = reward_function({'observation': observation['observation'],
                                            'achieved_goal': observation['achieved_goal'],
                                            'desired_goal': observation_['achieved_goal']}, {
                                             'observation': observation_['observation'],
                                             'achieved_goal': observation_['achieved_goal'],
                                             'desired_goal': observation_['achieved_goal']}
                                             , 0)

                for _ in range(2):
                    agent.remember(np.concatenate((observation['observation'], observation['achieved_goal'])), action, her_reward, np.concatenate(
                        (observation_['observation'], observation_['achieved_goal'])), True)

            if not load_checkpoint:
                agent.learn()

            score += reward
            observation = observation_.copy()
            obs_goal = obs_goal_.copy()

        if not load_checkpoint and i > 20:
            agent.plot_loss_value()
            for _ in range(50):
                agent.learn()
        score_history.append(score)
        avg_score = np.mean(score_history[-20:])
        long_avg_score = np.mean(score_history[-100:])

        if avg_score > long_avg_score and not load_checkpoint:
            agent.save_models()
        print('episode {} score {:.1f} avg score {:.1f}'.format(i, score, long_avg_score))

        if i % 50 == 0:

            env.close()
            env = gym.make('PandaPush-v3', render=True)
            score = 0
            for i_ in range(5):
                observation, info = env.reset()
                env.render()  # wait the right amount of time to make the rendering real-time
                obs_goal = np.concatenate(
                    (observation['observation'], observation['desired_goal']))
                obs_goal = normalizer.transform([obs_goal])[0]
                done = bool(info['is_success'])
                while not done:
                    action = agent.choose_action(obs_goal, is_eval=True)
                    observation, reward, terminated, truncated, info = env.step(
                        action.numpy())
                    env.render()
                    obs_goal = np.concatenate(
                        (observation['observation'], observation['desired_goal']))
                    obs_goal = normalizer.transform([obs_goal])[0]
                    done = terminated or truncated or bool(info['is_success'])
                    score += reward

            env.close()
            env = gym.make('PandaPush-v3', render=False)
            print('#####################################################')
            print('# evaluation {} average evaluation reward is: {:.1f} #'.format(
                np.uint(i / 50), score/5))
            print('#####################################################')

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, filename)


    env.close()

def main():
    print('Hi from nav_rl.')


if __name__ == '__main__':
    main()
