import gymnasium as gym
import random as rd
from game import game 
from gymnasium import Env, spaces 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging 
import sys
import os.path
from datetime import datetime
DTTM_FORMAT = '%Y%m%d%H%M%S'

## Idea
# We know from experimentation that only 10% of the games are completed 
# when two random players play. rest are drawn. 
# A good good is a game which is not draw.
# I want to train with 90 good games and 10 draw games Ruffly. Total 100 every time.
# Will do this 10 times and update the target model.
# So in Total 1000 Games , Trained every 100 games or so.
# in the 1000 Games I want 50% of exploitation and 50% of exploration.
# so 500 Games where we explore from 100% to 10% 
# Rest 500 Games where we explore 10% and exploit 90% 
# 500 Games = 500 * 25 Turns give or take = 12500 (X)moves in total
# Therefore from calculation 1 - X * (12500) = 0.1 we get 
# epsilon decay rate , X =  0.000072


# Global Variables 

# Configurations for Training
# Discount factor gamma, High discount factor since rewards
# appear only in the end.  
gamma = 0.85

# Epsilon greedy parameter
min_epsilon = 0.1
max_epsilon = 0.7
epsilon = max_epsilon  # starts with 1 slowly goes down to 0.1
epsilon_decay_factor = 0.000072

# batch size for training
batch_size = int(32*32) # Good initial choice
timesToSample =  int(900/32) # 600 * 32 , Ruffly 64  games out of 100 ?


# controlled within the game tbh
max_steps_per_episode = 999999

# Train the model after X games
update_after_games = 100

# How often to update the target network
update_target_network = 1000

# Good enough
max_memory_length = update_after_games * 300 * 2

# Using huber loss for stability
loss_function = keras.losses.Huber()
# Not sure what's happening here...
# will need to read more about this though.

# Optimizer - Adam Optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Arrays
action_history = []
state_history = []
state_next_history = []
rewards_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0

# save weights location 
saved_weights_dir = 'C:/w/ttu/RL/';
file_name = 'model'
file_extension = '.h5'
saved_weights_location = saved_weights_dir + file_name + file_extension

def main():
    # Initialize the Game Environment
    env = game()

    # Fetch the Observation space shape 
    # and num of actions from env
    state_shape = env.observation_space.shape
    total_actions = env.action_space.n
    print(' state_shape = ',state_shape, ' action_space = ',total_actions)

    # Create model and target model
    model = create_q_model(state_shape, total_actions)
    # for the sake of consistency 
    model_target = create_q_model(state_shape, total_actions)

    if os.path.exists(saved_weights_location):
        model.load_weights(saved_weights_location)
        model_target.load_weights(saved_weights_location)
        print('weights Loaded!!')

    # Let the Training Begin
    total_games = 0
    times_epsilon = 0
    epsilon = max_epsilon
    while True: # Will break when trained.
        
        # Reset Exploration and Exploitation
        if epsilon < 0.11:
            times_epsilon += 1
        
        if times_epsilon%4 == 0:
            times_epsilon = 0
            epsilon = max_epsilon

        # set running reward = 0
        running_reward = 0

        # update target model
        tn = int(update_target_network/update_after_games)
        for i in range(tn):
            good_games = 0
            running_reward = 0
            while(update_after_games != good_games):

                # Start a new Game.
                state = np.array(env.reset())
                episode_reward = 0
                action_history_now = []
                state_history_now = []
                state_next_history_now = []
                rewards_history_now = []
                episode_reward_history_now = []
                done = False
                # Complete the new game that is started.
                while done == False:
                    if epsilon > np.random.rand(1)[0]:
                        # Take random action
                        action = np.random.choice(total_actions)
                    else:
                        # Predict action Q-values
                        # From environment state
                        state_tensor = tf.convert_to_tensor(state)
                        state_tensor = tf.expand_dims(state_tensor, 0)
                        action_probs = model(state_tensor, training=False)
                        # Take best action
                        action = tf.argmax(action_probs[0]).numpy()
                    
                    epsilon -= epsilon_decay_factor
                    epsilon = max(min_epsilon , epsilon)

                    state_next, reward, done, info = env.step(action)
                    state_next = np.array(state_next)

                    # Save actions and states in replay buffer
                    action_history_now.append(action)
                    state_history_now.append(state)
                    state_next_history_now.append(state_next)
                    rewards_history_now.append(reward)
                    episode_reward = reward

                    state = state_next
                
                # Game is now completed. Check if its a good game.
                isGoodGame = False
                if episode_reward == 0:
                    #game is a draw and bad game, we need only 10 of these so lets do this.
                    if np.random.rand(1)[0] <= 0.40:
                        # lucky game will be added.
                        isGoodGame = True
                elif episode_reward == 100 or episode_reward == -100:
                    isGoodGame = True
                else:
                    print('Something wrong is happening.. check your memory and everything else in between please')
                    return
                
                if isGoodGame:
                    total_games += 1
                    good_games += 1
                    # moving average , pretty cool math
                    running_reward = (episode_reward + (good_games-1) * running_reward ) / good_games
                    for i in range(len(rewards_history_now)):
                        action_history.append(action_history_now[i])
                        state_history.append(state_history_now[i])
                        state_next_history.append(state_next_history_now[i])
                        rewards_history.append(rewards_history_now[i])
                
                # That's it we can end the game now
            
            # Done with good number of games. Time to Train. Lets go!!!
            # 30000 -> timesToSample
            # len(action_history) -> ?
            timesToSample_now = int(timesToSample * len(action_history) / 30000) + 1
            for i in range(timesToSample_now):
                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(rewards_history_now)), size=batch_size)

                # Get samples for training
                state_sample = np.array([state_history[i] for i in indices])
                state_next_sample = np.array([state_next_history[i] for i in indices])
                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]

                # Build the Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(state_next_sample)

                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, total_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)
                
                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # clear up memory
            if len(rewards_history) > max_memory_length:
                print('Clearing half memory')
                #del a[:int(len(aasdas)/2)]
                del rewards_history[:int(len(rewards_history)/2)]
                del state_history[:int(len(state_history)/2)]
                del state_next_history[:int(len(state_next_history)/2)]
                del action_history[:int(len(action_history)/2)]
            
            # Log details
            print()
            template = "running reward: {:.2f} at game {} , epsilon = {} , memory = {}"
            print(template.format(running_reward, total_games, epsilon,len(rewards_history)))
            print()

            #saving the model
            model.save_weights(saved_weights_location)

        
        # Time to Update the Target Model
        # update the the target network with new weights
        model_target.set_weights(model.get_weights())

        # Log details
        print()
        template = "running reward: {:.2f} at game {} , epsilon = {} , memory = {}"
        print(template.format(running_reward, total_games, epsilon,len(rewards_history)))
        print()

        #saving the model
        model.save_weights(saved_weights_dir + file_name + '_' +str(total_games)+ '_' + str(datetime.now().strftime(DTTM_FORMAT)) + file_extension)
        
        # if running reward is 70 or more, with 10% random exploration then its good right ? 
        if running_reward > 70:
            break


# Networks
def create_q_model(state_shape, total_actions):
    # input layer
    inputs = layers.Input(shape=state_shape)

    # Hidden layers
    layer1 = layers.Dense(64, activation="relu")(inputs)
    layer2 = layers.Dense(72, activation="relu")(layer1)
    layer3 = layers.Dense(64, activation="relu")(layer2)

    # output layer    
    action = layers.Dense(total_actions, activation="linear")(layer3)

    return keras.Model(inputs=inputs, outputs=action)


if __name__=="__main__":
    main()