import gymnasium as gym
import random as rd
from game_ma import game 
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


# Global Variables - Hyper Parameters

# Configurations for Training
# Discount factor gamma, High discount factor since rewards
# appear only in the end.  
gamma = 0.9

# Epsilon greedy parameter
# min and max are very low because of very high illegal rate of the game.
# 20 games 0.1 -> 0.0 Max is 3100 * 20 Min is 15 * 20 avg is  31150 31012.5
min_epsilon = 0.05
#min_epsilon = 0.005
max_epsilon = 0.25
#max_epsilon = 0.015
epsilon = max_epsilon  # slowly goes down to min_epsilon
#pre training
epsilon_decay_factor = 0.000000333
#epsilon_decay_factor = 0.0000148
#epsilon_decay_factor = 0.0000198
#epsilon_decay_factor = 0.0000064
epsilon_reset_after = 1
epsilon_check_after_games = 300 

# batch size for training
batch_size = int(32*200) # Good initial choice
timesToSample =  12   # sampling 4 times 
min_memory_size = 32*200

#max_steps_per_episode = Controlled within the game.

# Train the model after X games
update_after_games = 10

# How often to update the target network
update_target_network = 500

# Good enough
max_memory_length = 100000

# Using huber loss for stability
loss_function = keras.losses.Huber()
# Not sure what's happening here...
# will need to read more about this though.
# read about it, best of RMS and other Absoulte loss funciton. 

# Optimizer - Adam Optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Memory
# [state, action, reward, next_state, done]
replay_buffer_state = [];
replay_buffer_action = [];
replay_buffer_reward = [];
replay_buffer_next_state = [];
replay_buffer_done = [];

buffer_state_local = [];
buffer_action_local = [];
buffer_reward_local = [];
buffer_next_state_local = [];
buffer_done_local = [];


buffer_state_ts = [];
buffer_action_ts = [];
buffer_reward_ts = [];
buffer_next_state_ts = [];
buffer_done_ts = [];


running_reward = 0
episode_count = 0

# save weights location 
saved_weights_dir = './model/';
file_name = 'model'
file_extension = '.h5'
saved_weights_location = saved_weights_dir + file_name + file_extension
save_weights_after = 600


def main():
    # Initialize the Game Environment
    #env = game()
    env = game('./assistance_model/model.h5')
    #log
    total_wins = 0
    total_loss = 0
    total_draws = 0
    min_epsilon = 0.05

    #draw_percentage_games = 0.15
    draw_percentage_games = 2.0
    win_all_time = 1000
    agent_training_completed = 0

    # Fetch the Observation space shape 
    # and num of actions from env
    state_shape = env.observation_space.shape
    total_actions = env.action_space.n
    print(' state_shape = ',state_shape, ' action_space = ',total_actions)

    # Create model and target model
    model = create_q_model(state_shape, total_actions)
    # for the sake of consistency 
    model_target = create_q_model(state_shape, total_actions)
    model.summary()

    if os.path.exists(saved_weights_location):
        model.load_weights(saved_weights_location)
        model_target.load_weights(saved_weights_location)
        print('weights Loaded!!')

    # Let the Training Begin
    total_games = 0
    times_epsilon = 1
    epsilon = max_epsilon
    while agent_training_completed <= win_all_time: # Will break when trained.
        
        # Reset Exploration and Exploitation
        #if epsilon == min_epsilon:
        #    times_epsilon += 1
        
        #if times_epsilon%epsilon_reset_after == 0:
        #    times_epsilon = 1
        #    epsilon = max_epsilon

        # set running reward = 0
        running_reward = 0
        
        min_epsilon = np.random.choice([0.0,0.05],1,p=[0.5,0.5])[0]

        # update target model
        tn = int(epsilon_check_after_games/update_after_games)
        for i in range(tn):
            epsilon_target = int(tn - 17)
            epsilon = (0.7 - (i*0.7/epsilon_target))
            if(epsilon < 0):
                epsilon = 0
            good_games = 0
            running_reward = 0
            actions_taken = 0
            buffer_state_ts = [];
            buffer_action_ts = [];
            buffer_reward_ts = [];
            buffer_next_state_ts = [];
            buffer_done_ts = [];
            while(update_after_games != good_games):

                # Start a new Game.
                #state = np.array(env.reset())
                
                state = env.reset()
                episode_reward = 0
                done = False
                local_actions_taken = 0
                # Complete the new game that is started.
                buffer_state_local = [];
                buffer_action_local = [];
                buffer_reward_local = [];
                buffer_next_state_local = [];
                buffer_done_local = [];
                while done == False:
                    action = -1
                    if epsilon > np.random.rand(1)[0]:
                        # Take random action
                        #print('before', len(buffer_done_local))
                        isValidActionTaken = False
                        while isValidActionTaken == False:
                            action = np.random.choice(total_actions)
                            isValidActionTaken = env.isValidAction(action)
                            if isValidActionTaken == False:
                                buffer_state_local.append(state);
                                buffer_action_local.append(action);
                                buffer_reward_local.append(-1);
                                buffer_next_state_local.append(state);
                                buffer_done_local.append(True); 
                                #print('during' , len(buffer_done_local))
                        #print('after', len(buffer_done_local))
                            
                    else:
                        # Predict action Q-values
                        # From environment state
                        state_tensor = tf.convert_to_tensor(state)
                        state_tensor = tf.expand_dims(state_tensor, 0)
                        action_probs = model(state_tensor, training=False)
                        # Take best action
                        action = tf.argmax(action_probs[0]).numpy()
                        isValidActionTaken = False
                        while isValidActionTaken == False:
                            isValidActionTaken = env.isValidAction(action)
                            if isValidActionTaken == False:
                                buffer_state_local.append(state);
                                buffer_action_local.append(action);
                                buffer_reward_local.append(-1);
                                buffer_next_state_local.append(state);
                                buffer_done_local.append(True); 
                                # Take a random action
                                action = np.random.choice(total_actions)
                        
                    
                    # epsilon greedy stuff..
                    epsilon -= epsilon_decay_factor
                    epsilon = max(min_epsilon , epsilon)

                    state_next, reward, done, info = env.step(action)
                    

                    # Save actions and states in replay buffer
                    #replay_buffer_local.append([state,action,reward,state_next,done])
                    buffer_state_local.append(state);
                    buffer_action_local.append(action);
                    buffer_reward_local.append(reward);
                    buffer_next_state_local.append(state_next);
                    buffer_done_local.append(done);
                    #action_history_now.append(action)
                    #state_history_now.append(state)
                    #state_next_history_now.append(state_next)
                    #rewards_history_now.append(reward)
                    #done_history_now.append(done)
                    episode_reward = reward
                    
                    #learn from mistakes ?
                    if done == True and reward == -1 :
                        zstate = np.copy(state)
                        zstate_next = np.copy(state_next)
                        buffer_state_local.append(zstate*-1);
                        buffer_action_local.append(int(info['random_action']));
                        buffer_reward_local.append(1);
                        buffer_next_state_local.append(zstate_next*-1);
                        buffer_done_local.append(done);
                        #print(buffer_state_local[-2])
                        #print(buffer_action_local[-2])
                        #print(buffer_reward_local[-2])
                        #print(buffer_next_state_local[-2])
                        #print(buffer_done_local[-2])
                        #print(buffer_state_local[-1])
                        #print(buffer_action_local[-1])
                        #print(buffer_reward_local[-1])
                        #print(buffer_next_state_local[-1])
                        #print(buffer_done_local[-1])
                        
                        

                    state = state_next
                    local_actions_taken += 1
                
                # Game is now completed.                   
                # Check if its a good game.
                isGoodGame = False
                if episode_reward == 0:
                    #game is a draw and a bad game, we need only 10% of these so lets do this.
                    if np.random.rand(1)[0] <= draw_percentage_games:
                        # lucky game will be added.
                        isGoodGame = True
                else:
                    isGoodGame = True
                       
                if isGoodGame:
                    total_games += 1
                    good_games += 1
                    actions_taken += local_actions_taken
                    # moving average , pretty cool math
                    running_reward = (episode_reward + (good_games-1) * running_reward ) / good_games
                    
                    # logging more data
                    if(episode_reward == 0 or episode_reward == -0.05):
                        total_draws += 1
                    elif(episode_reward == 1):
                        total_wins += 1
                        with open("./data/winning_moves.txt", "a")  as f:
                            print("", file=f)
                            print(buffer_state_local[-1] , file=f)
                            print(buffer_action_local[-1] , file=f)
                            print(buffer_reward_local[-1] ,  file=f)
                            print(np.reshape(buffer_next_state_local[-1], (4, 4)) , file=f)
                            print("", file=f)
                    elif(episode_reward == -1):
                        total_loss += 1
                        with open("./data/lost_moves.txt", "a")  as f:
                            print("", file=f)
                            print(buffer_state_local[-1] , file=f)
                            print(buffer_action_local[-1] , file=f)
                            print(buffer_reward_local[-1] ,  file=f)
                            print(np.reshape(buffer_next_state_local[-1], (4, 4)) , file=f)
                            print("", file=f)
                        
                    
                    for lrbl in range(len(buffer_action_local)):
                        # update the training set
                        buffer_state_ts.append(buffer_state_local[lrbl])
                        buffer_action_ts.append(buffer_action_local[lrbl])
                        buffer_reward_ts.append(buffer_reward_local[lrbl])
                        buffer_next_state_ts.append(buffer_next_state_local[lrbl])
                        buffer_done_ts.append(buffer_done_local[lrbl])
                        # update replay buffer
                        replay_buffer_state.append(buffer_state_local[lrbl])
                        replay_buffer_action.append(buffer_action_local[lrbl])
                        replay_buffer_reward.append(buffer_reward_local[lrbl])
                        replay_buffer_next_state.append(buffer_next_state_local[lrbl])
                        replay_buffer_done.append(buffer_done_local[lrbl])

                        
                        
                
            # That's it we can end the game now
            
            # Done with good number of games. Time to Train. Lets go!!!
            # 30000 -> timesToSample XXX
            # len(action_history) -> ? XXXX
            # total_actions = 300000 -> timesToSample
            # total_actions -> ?
            #timesToSample_now = int(timesToSample * total_actions / 15000) + 2
            #timesToSample_now = min(timesToSample, timesToSample_now)
            
            for _ in range(timesToSample):
                #break;
                # Get indices of samples for replay buffers
                if(len(replay_buffer_action) > min_memory_size):

                    # get random order from training set.
                    indices_training_order = np.arange(len(buffer_action_ts))
                    np.random.shuffle(indices_training_order)
                    # for each random move from training memory.
                    #print("indices_training_order")
                    #print(indices_training_order)
                    
                    #for ito in indices_training_order:
                    #    pass

                    # get batchsize -1 random moves from replay buffer
                    indices_replay_buffer = np.random.choice(range(len(replay_buffer_action)), size=(batch_size-len(buffer_action_ts)))
                    #print("indices_replay_buffer")
                    #print(indices_replay_buffer)

                    # separate things for training.
                    state_sample = np.array([replay_buffer_state[td] for td in indices_replay_buffer])
                    action_sample = np.array([replay_buffer_action[td] for td in indices_replay_buffer])
                    reward_sample = np.array([replay_buffer_reward[td] for td in indices_replay_buffer])
                    state_next_sample = np.array([replay_buffer_next_state[td] for td in indices_replay_buffer])
                    done_sample = np.array([replay_buffer_done[td] for td in indices_replay_buffer])
                    # print("state_next_sample")
                    # print(state_next_sample.shape)
                    # state_sample = np.append(state_sample, np.array([buffer_state_ts[ito]]), axis=0)
                    # action_sample = np.append(action_sample, np.array([buffer_action_ts[ito]]), axis=0)
                    # reward_sample = np.append(reward_sample, np.array([buffer_reward_ts[ito]]), axis=0)
                    # state_next_sample = np.append(state_next_sample, np.array([buffer_state_ts[ito]]), axis=0)
                    # done_sample = np.append(done_sample, np.array([buffer_done_ts[ito]]), axis=0)

                    state_sample = np.append(state_sample, np.array(buffer_state_ts), axis=0)
                    action_sample = np.append(action_sample, np.array(buffer_action_ts), axis=0)
                    reward_sample = np.append(reward_sample, np.array(buffer_reward_ts), axis=0)
                    state_next_sample = np.append(state_next_sample, np.array(buffer_state_ts), axis=0)
                    done_sample = np.append(done_sample, np.array(buffer_done_ts), axis=0)

                    indices_sample_order = np.arange(len(state_sample))
                    np.random.shuffle(indices_sample_order)
                    state_sample = state_sample[indices_sample_order]
                    action_sample = action_sample[indices_sample_order] 
                    reward_sample = reward_sample[indices_sample_order]
                    state_next_sample = state_next_sample[indices_sample_order]
                    done_sample = done_sample[indices_sample_order]
                    

                    # print("state_next_sample")
                    # print(state_next_sample.shape)
                    #print("train_data")
                    #print(train_data)
                    # print("action_sample")
                    # print(action_sample)
                    # print("reward_sample")
                    # print(reward_sample)
                    #print("state_next_sample")
                    #print(state_next_sample)
                    #print(state_next_sample.shape)
                    # print("done_sample")
                    # print(done_sample)

                    # Build the Q-values for the sampled future states
                    # Use the target model for stability
                    future_rewards = model_target.predict(state_next_sample)

                    updated_q_values = reward_sample + gamma * tf.reduce_max(
                        future_rewards, axis=1
                    )
                    #print("updated_q_values")
                    #print(updated_q_values)
                    new_update_q_values = updated_q_values.numpy()
                    # If final frame set the last value to -1
                    #updated_q_values = updated_q_values * (1 - done_sample) - done_sample
                    for ildse in range(len(done_sample)):
                            if(done_sample[ildse]):
                                if(int(reward_sample[ildse]) != 0):
                                    new_update_q_values[ildse] = reward_sample[ildse]
                    updated_q_values = tf.convert_to_tensor(new_update_q_values)
                    #print("updated_q_values")
                    #print(updated_q_values)
                    

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
                    #print(loss)
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    #input()

            # clear up memory
            if len(replay_buffer_action) > max_memory_length:
                print('freeing up replay buffer memory')
                #del a[:int(len(aasdas)/2)]
                #del replay_buffer[:int(len(replay_buffer)/5)]
                del replay_buffer_action[:int(len(replay_buffer_action)/5)]
                del replay_buffer_done[:int(len(replay_buffer_done)/5)]
                del replay_buffer_next_state[:int(len(replay_buffer_next_state)/5)]
                del replay_buffer_reward[:int(len(replay_buffer_reward)/5)]
                del replay_buffer_state[:int(len(replay_buffer_state)/5)]
            
            # if running reward is 70 or more, with 10% random exploration then its good right ? 
            if running_reward > 0.9:
                agent_training_completed += 1
                
                
            
            # Log details
            print()
            template = "running reward: {:.2f} at game {} ({}) , epsilon = {:.3f} , memory = {}, actions = {} , {}"
            template2 = "total_draws: {}, total_wins: {}, total_loss: {}"
            print(template.format(running_reward, total_games, (i+1), epsilon,len(replay_buffer_action), len(buffer_action_ts), datetime.now()))
            print(template2.format(total_draws,total_wins, total_loss), flush=True)
            print()

            #saving the model
            model.save_weights(saved_weights_location)

        
        # Time to Update the Target Model
        # update the the target network with new weights
        if(total_games%update_target_network == 0):
            model_target.set_weights(model.get_weights())

        # Log details
        print()
        template = "running reward: {:.2f} at game {} , epsilon = {:.3f} , memory = {}, completed = {:.2f}%"
        print(template.format(running_reward, total_games, epsilon,len(replay_buffer_action), (agent_training_completed*100/win_all_time)), flush=True)
        print()

        #saving the model
        if total_games%save_weights_after == 0:
            model.save_weights(saved_weights_dir + file_name + '_' +str(total_games)+ '_' + str(datetime.now().strftime(DTTM_FORMAT)) + file_extension)


# Networks
def create_q_model(state_shape, total_actions):
    # input layer
    inputs = layers.Input(shape=state_shape)

    # Hidden layers
    layer1 = layers.Dense(500, activation="relu")(inputs)
    layer2 = layers.Dense(250, activation="relu")(layer1)
    #layer3 = layers.Dense(1000, activation="relu")(layer2)
    #layer4 = layers.Dense(45, activation="relu")(layer3)   
    #layer5 = layers.Dense(10, activation="relu")(layer4)

    # output layer    
    action = layers.Dense(total_actions, activation="linear")(layer2)

    return keras.Model(inputs=inputs, outputs=action)


if __name__=="__main__":
    main()
