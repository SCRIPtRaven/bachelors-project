import random
from collections import deque

import numpy as np
from keras import Input, Model
from keras.src.layers import Add, Subtract, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DQNAgent:
    """
    Basic DQN agent for disruption resolution
    """

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = 0.99  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.learning_rate = 0.0005
        self.tau = 0.001  # For soft target updates

        self.memory = deque(maxlen=50000)

        self.priorities = deque(maxlen=50000)
        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        self.priority_beta_increment = 0.001
        self.priority_epsilon = 0.01

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Build a dueling DQN network with improved architecture"""
        input_layer = Input(shape=(self.state_size,))

        shared = Dense(128, activation='relu')(input_layer)
        shared = Dense(128, activation='relu')(shared)

        value_stream = Dense(64, activation='relu')(shared)
        value = Dense(1)(value_stream)

        advantage_stream = Dense(64, activation='relu')(shared)
        advantage = Dense(self.action_size)(advantage_stream)

        q_values = Add()([value, Subtract()([advantage,
                                             Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(advantage)])])

        model = Model(inputs=input_layer, outputs=q_values)
        model.compile(loss='huber', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Soft update of target network"""
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]

        self.target_model.set_weights(target_weights)

    def remember(self, state, action, reward, next_state, done):
        """Store experience with priority"""
        priority = abs(reward) + self.priority_epsilon

        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(priority)

    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """Train the model using prioritized experience replay"""
        if len(self.memory) < batch_size:
            return

        priorities = np.array(self.priorities)
        probabilities = priorities ** self.priority_alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        batch = [self.memory[i] for i in indices]

        self.priority_beta = min(1.0, self.priority_beta + self.priority_beta_increment)
        weights = (len(self.memory) * probabilities[indices]) ** (-self.priority_beta)
        weights /= weights.max()

        states = np.array([item[0] for item in batch])
        actions = np.array([item[1] for item in batch])
        rewards = np.array([item[2] for item in batch])
        next_states = np.array([item[3] for item in batch])
        dones = np.array([item[4] for item in batch])

        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)

        target_q_values = self.target_model.predict(next_states, verbose=0)
        target_values = np.array([target_q_values[i, next_actions[i]]
                                  for i in range(batch_size)])

        targets = rewards + self.gamma * target_values * (1 - dones)

        current_q = self.model.predict(states, verbose=0)

        for i in range(batch_size):
            current_q[i, actions[i]] = targets[i]

        self.model.fit(states, current_q, sample_weight=weights, epochs=1, verbose=0)

        for i, idx in enumerate(indices):
            td_error = abs(targets[i] - self.model.predict(states[i].reshape(1, -1), verbose=0)[0, actions[i]])
            self.priorities[idx] = td_error + self.priority_epsilon

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Load model weights from file"""
        self.model.load_weights(name)

    def save(self, name):
        """Save model weights with proper Keras format"""
        # Ensure filename ends with .weights.h5
        if not name.endswith('.weights.h5'):
            if name.endswith('.h5'):
                name = name.replace('.h5', '.weights.h5')
            else:
                name = name + '.weights.h5'
        self.model.save_weights(name)
