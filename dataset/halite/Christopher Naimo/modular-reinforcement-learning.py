"""
This kernel is a more modular approach to the notebook "Designing game AI with Reinforcement learning" by Victor Basu. 
The objective here was to modify his notebook into a script which could utilize multiple actor/critic models simultaneously.
DocStrings have been included for clarity. I'm running this locally on linux with TensorFlow GPU v1.14.


I hope this is helpful for those who are exploring reinforcement learning for this year's halite competition. Good luck!


UPDATES FROM V3->V4:
    -refactoring
    -updated DocStrings
    -new ship agent
    -model trains much faster
    -reward vs. episode plot
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import preprocessing
from matplotlib import pyplot as plt
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *

import logging
import os
import sys

import numpy as np  
import tensorflow as tf
from tqdm import tqdm

tf.enable_eager_execution() # required for TensorFlow v1.14


class LOGIC:
    def __init__(self, labels: List[str], agent: Callable, board_converter: Callable, rl_model: tf.keras.Model):
        """
        Class for handling a neural net agent. Includes: training, next action generation, and more!

        :param labels: string names for possible actions of this agent
        :param agent: agent function
        :param board_converter: function to convert board to neural net input
        :param rl_model: neural net model
        """
        self.optimizer = tf.keras.optimizers.Adam(lr=7e-4)
        self.huber_loss = tf.keras.losses.Huber()
        self.action_probs_history = list()
        self.critic_value_history = list()
        self.rewards_history = list()
        self.running_reward = 0
        self.episode_count = 0
        self.num_actions = 5
        self.eps = np.finfo(np.float32).eps.item()
        self.gamma = 0.99  # Discount factor for past rewards
        self.le = preprocessing.LabelEncoder()
        self.label_encoded = self.le.fit_transform(labels)
        self.agent = agent
        self.model = rl_model
        self.convert_board = board_converter

    def train_step(self, current_board: Board, ship_index: int) -> Union[ShipAction, ShipyardAction]:
        """
        Train model for one time step with provided game board

        :param current_board: Board object
        :param ship_index: index of ship or shipyard
        :return: next action
        """

        model_input = self.convert_board(current_board)
        action_prob, critic_value = self.model(model_input)
        self.critic_value_history.append(critic_value[0, 0])
        current_action = np.random.choice(self.num_actions, p=action_prob.numpy()[0])
        self.action_probs_history.append(tf.math.log(action_prob[0, current_action]))
        current_action = self.le.inverse_transform([current_action])[0]
        return self.agent(board, current_action, ship_index)

    def add_gain(self, step_gain: int) -> None:
        """
        append step gain to model

        :param step_gain: step gain
        :return: None
        """
        self.rewards_history.append(step_gain)

    def propagate(self, gradient_tape: tf.GradientTape) -> None:
        """
        Manage reward calculation & back-propagation through network

        :param gradient_tape: TensorFlow gradient tape
        :return: None
        """

        # Calculate expected value from rewards
        # - At each time step what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0

        for r in self.rewards_history[::-1]:
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
        returns = returns.tolist()
        # Calculating loss values to update our network
        history = zip(self.action_probs_history, self.critic_value_history, returns)

        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up receiving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )
        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = gradient_tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Clear the loss and reward history
        self.action_probs_history.clear()
        self.critic_value_history.clear()
        self.rewards_history.clear()

    def get_action(self, current_board: Board, ship_index: int) -> Union[ShipAction, ShipyardAction]:
        """
        Generate next action

        :param current_board: Board object
        :param ship_index: index of ship or shipyard
        :return: next action
        """
        model_input = self.convert_board(current_board)
        action_prob, _ = self.model(model_input)
        current_action = np.random.choice(self.num_actions, p=action_prob.numpy()[0])
        current_action = self.le.inverse_transform([current_action])[0]
        return self.agent(board, current_action, ship_index)


seed = 123
tf.set_random_seed(seed)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
logging.disable(sys.maxsize)
global ship_


def actor_model(num_actions, in_):
    common = tf.keras.layers.Dense(128, activation='tanh')(in_)
    common = tf.keras.layers.Dense(32, activation='tanh')(common)
    common = tf.keras.layers.Dense(num_actions, activation='softmax')(common)
    return common


def critic_model(in_):
    common = tf.keras.layers.Dense(128)(in_)
    common = tf.keras.layers.ReLU()(common)
    common = tf.keras.layers.Dense(32)(common)
    common = tf.keras.layers.ReLU()(common)
    common = tf.keras.layers.Dense(1)(common)
    return common


input_ = tf.keras.layers.Input(shape=[441, ])
model = tf.keras.Model(inputs=input_, outputs=[actor_model(5, input_), critic_model(input_)])
print(model.summary())

running_reward = 0
episode_count = 0

env = make("halite", debug=True)
trainer = env.train([None, "random"]) # you may have to specify a python file for 'random'


def get_dir_to(from_pos, to_pos, size):
    from_x, from_y = divmod(from_pos[0], size), divmod(from_pos[1], size)
    to_x, to_y = divmod(to_pos[0], size), divmod(to_pos[1], size)
    if from_y < to_y:
        return ShipAction.NORTH
    if from_y > to_y:
        return ShipAction.SOUTH
    if from_x < to_x:
        return ShipAction.EAST
    if from_x > to_x:
        return ShipAction.WEST


# Directions a ship can move
directions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]


def decode_dir(act_: str) -> Union[ShipAction, None]:
    """
    Get ShipAction from string

    :param act_: string action
    :return: ShipAction
    """

    decode = {
        'NORTH': ShipAction.NORTH,
        'EAST': ShipAction.EAST,
        'WEST': ShipAction.WEST,
        'SOUTH': ShipAction.SOUTH,
        'CONVERT': ShipAction.CONVERT,
        'NONE': None
    }
    return decode[act_]




def advanced_agent(board: Board, action: str, ship_index: int):
    # Returns the commands we send to our ships and shipyards
    me = board.current_player
    act = action

    if act == "CONVERT" and len(me.ships) / (1 if len(me.shipyards) == 0 else len(me.shipyards)) < 3 and me.ships:
        # minimum 3 ships per shipyard
        me.ships[ship_index].next_action = None
        return me.next_actions

    # If there are no ships, use first shipyard to spawn a ship.
    if len(me.ships) == 0 and len(me.shipyards) > 0:
        me.shipyards[0].next_action = ShipyardAction.SPAWN
        return me.next_actions

    # If there are no shipyards, convert first ship into shipyard.
    if len(me.shipyards) == 0 and len(me.ships) > 0:
        me.ships[ship_index].next_action = ShipAction.CONVERT
    elif len(me.ships) > 0:
        if me.ships[ship_index].halite > 200:
            direction = get_dir_to(me.ships[0].position, me.shipyards[0].position, board.configuration.size)
            if direction:
                me.ships[0].next_action = direction
        else:
            me.ships[ship_index].next_action = decode_dir(act)
    return me.next_actions


def convert(board: Board) -> tf.Tensor:
    """
    Extract relevant board/player data and convert to tensor input

    :param board: Board object
    :return: tensor
    """
    state_ = tf.convert_to_tensor([board.cells[Point(x, y)].halite for x in range(21) for y in range(21)])
    state_ = tf.expand_dims(state_, 0)
    return state_


ship_model = LOGIC(labels=['NORTH', 'SOUTH', 'EAST', 'WEST', 'CONVERT', 'NONE'], agent=advanced_agent,
                   board_converter=convert,
                   rl_model=model)


training = list() # logs reward for each episode

while not env.done:
    state = trainer.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in tqdm(range(1, env.configuration.episodeSteps + 200)):
            board = Board(state, env.configuration)
            action = ship_model.train_step(board, 0)
            state = trainer.step(action)[0]
            gain = state.players[0][0] / 5000
            ship_model.add_gain(gain)
            episode_reward += gain
            if env.done:
                state = trainer.reset()
                # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        training.append([episode_count, running_reward])
#         print("reward:", running_reward)
        ship_model.propagate(tape)

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 550:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break
#     if episode_count >= 3:
#         print("max episode reached, training complete!")
#         break

# plot reward vs training episode
plt.plot([x[0] for x in training], [x[1] for x in training])
plt.show()


"""
I use this to generate the halite simulation and run automatically using firefox webdriver, optional


from selenium import webdriver
out = env.render(mode="html", width=800, height=600)
# Write the output to a html file so we can open in a browser.
f = open("halite.html", "w")
f.write(out)
f.close()
# return
driver = webdriver.Firefox()
html_file = os.getcwd() + "//" + "halite.html"
driver.get("file:///" + html_file)
"""