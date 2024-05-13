import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, SGD
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, alpha, beta, input_dims, n_actions,
                 h_actions, l_actions, tau,
                 gamma=0.9, update_actor_interval=2, warmup=1000,
                 max_size=1000000, layer1_size=128, layer2_size=256, layer3_size=64, batch_size=128, noise=0.1,
                 chkpt_dir='H:\\akari103\\Desktop\\tf2models\\'):
        self.gamma = gamma
        self.tau = tau
        self.max_action = h_actions
        self.min_action = l_actions
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.fname = chkpt_dir
        self.update_actor_iter = update_actor_interval
        self.save_permission = False

        self.loss = np.array([[0, 0]])
        self.value = np.array([[0, 0]])
        self.ax = plt.gca()
        plt.ion()
        self.ax.set(ylim=(-1.5, 1.5))
        plt.show()
        plt.pause(0.01)

        self.actor = ActorNetwork(layer1_size, layer2_size, layer3_size,
                                  n_actions=n_actions)

        self.critic_1 = CriticNetwork(layer1_size, layer2_size, layer3_size)
        self.critic_2 = CriticNetwork(layer1_size, layer2_size, layer3_size)

        self.target_actor = ActorNetwork(layer1_size, layer2_size, layer3_size,
                                         n_actions=n_actions)
        self.target_critic_1 = CriticNetwork(
            layer1_size, layer2_size, layer3_size)
        self.target_critic_2 = CriticNetwork(
            layer1_size, layer2_size, layer3_size)

        # ac_lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=alpha,
        #     decay_steps=4000,
        #     decay_rate=0.9985)
        cr_lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=beta,
            decay_steps=8000,
            decay_rate=0.99955)

        self.actor.compile(optimizer=SGD(learning_rate=alpha))
        self.critic_1.compile(optimizer=SGD(learning_rate=cr_lr_schedule))
        self.critic_2.compile(optimizer=SGD(learning_rate=cr_lr_schedule))

        self.target_actor.compile(optimizer=SGD(learning_rate=alpha))
        self.target_critic_1.compile(
            optimizer=SGD(learning_rate=cr_lr_schedule))
        self.target_critic_2.compile(
            optimizer=SGD(learning_rate=cr_lr_schedule))

        self.noise = noise
        self.update_network_parameters(tau=1)

    def save_models(self):
        if self.memory.mem_cntr > 10 * self.batch_size and self.save_permission and False:
            print('... saving models ...')
            self.actor.save(self.fname+'actor')
            self.critic_1.save(self.fname+'critic_1')
            self.critic_2.save(self.fname+'critic_2')
            self.target_actor.save(self.fname+'target_actor')
            self.target_critic_1.save(self.fname+'target_critic_1')
            self.target_critic_2.save(self.fname+'target_critic_2')

    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.fname+'actor')
        self.critic_1 = keras.models.load_model(self.fname+'critic_1')
        self.critic_2 = keras.models.load_model(self.fname+'critic_2')
        self.target_actor = keras.models.load_model(self.fname+'target_actor')
        self.target_critic_1 = \
            keras.models.load_model(self.fname+'target_critic_1')
        self.target_critic_2 = \
            keras.models.load_model(self.fname+'target_critic_2')

    def choose_action(self, observation, suggest_action=None, is_eval=False):
        if is_eval:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            return self.actor(state)[0]
        
        if self.time_step < self.warmup:
            if np.any(np.array(suggest_action) == None):
                mu = np.random.normal(scale=self.noise*5,
                                      size=(self.n_actions,))
            else:
                mu = 3*suggest_action + np.random.normal(
                    scale=self.noise*5, size=(self.n_actions,))
        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            # returns a batch size of 1, want a scalar array
            mu = self.actor(state)[0]

        mu_prime = mu + \
            np.random.normal(scale=self.noise, size=(self.n_actions,))
        mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)
        self.time_step += 1

        return mu_prime

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < 10 * self.batch_size:
            return

        states, actions, rewards, new_states, dones = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(states_)
            target_actions = target_actions + \
                tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)

            target_actions = tf.clip_by_value(target_actions, self.min_action,
                                              self.max_action)

            q1_ = self.target_critic_1((states_, target_actions))
            q2_ = self.target_critic_2((states_, target_actions))

            q1 = tf.squeeze(self.critic_1((states, actions)), 1)
            q2 = tf.squeeze(self.critic_2((states, actions)), 1)

            # shape is [batch_size, 1], want to collapse to [batch_size]
            q1_ = tf.squeeze(q1_, 1)
            q2_ = tf.squeeze(q2_, 1)

            critic_value_ = tf.math.minimum(q1_, q2_)
            # in tf2 only integer scalar arrays can be used as indices
            # and eager exection doesn't support assignment, so we can't do
            # q1_[dones] = 0.0
            target = rewards + self.gamma*critic_value_*(1-dones)
            critic_1_loss = keras.losses.MSE(target, q1)
            critic_2_loss = keras.losses.MSE(target, q2)
        params_1 = self.critic_1.trainable_variables
        # print(params_1)
        params_2 = self.critic_2.trainable_variables
        grads_1 = tape.gradient(critic_1_loss, params_1)
        grads_1 = [(tf.clip_by_norm(grad, 3)) for grad in grads_1]
        grads_2 = tape.gradient(critic_2_loss, params_2)
        grads_2 = [(tf.clip_by_norm(grad, 3)) for grad in grads_2]

        self.critic_1.optimizer.apply_gradients(zip(grads_1, params_1))
        self.critic_2.optimizer.apply_gradients(zip(grads_2, params_2))

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            critic_1_value = self.critic_1((states, new_actions))
            # actor_loss = 0.1 * tf.math.reduce_mean(
            #     tf.norm(new_actions, axis=1)) - tf.math.reduce_mean(critic_1_value)
            actor_loss = - tf.math.reduce_mean(critic_1_value)
        if critic_1_loss < 0.65 and self.learn_step_cntr > 10000:
            self.save_permission = True
            params = self.actor.trainable_variables
            # print(params)
            grads = tape.gradient(actor_loss, params)
            grads = [(tf.clip_by_norm(grad, 2)) for grad in grads]
            # print(grads, params)
            self.actor.optimizer.apply_gradients(zip(grads, params))
        else:
            self.save_permission = False

        self.loss = np.append(self.loss, [[critic_1_loss, 0]], axis=0)
        self.value = np.append(self.value, [[actor_loss, 0]], axis=0)
        if self.learn_step_cntr == 2:
            self.loss = self.loss[1:]
            self.value = self.value[1:]

        self.loss[-1,1] = np.average(self.loss[-101:, 0])
        self.value[-1,1] = np.average(self.value[-101:, 0])

        # if len(self.loss) > 10000:
        #     self.loss = self.loss[1:]
        #     self.value = self.value[1:]

        self.update_network_parameters()

    def plot_loss_value(self):
        #plt.axis([0, 10, 0, 1])
        self.ax.clear()
        self.ax.set(ylim=(0.1, 50))
        x = np.arange(np.shape(self.loss)[0])
        self.ax.plot(x, self.loss[:, 1], label="loss")
        self.ax.fill_between(
            x, self.loss[:, 0], self.loss[:, 1], alpha=.5, linewidth=0, label="loss_range")
        self.ax.plot(x, self.value[:, 1], label="-value")
        self.ax.fill_between(
            x, self.value[:, 0], self.value[:, 1], alpha=.5, linewidth=0, label="-value_range")
        plt.legend()
        plt.yscale("log")
        plt.draw()
        plt.pause(1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_critic_1.set_weights(weights)

        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_critic_2.set_weights(weights)
