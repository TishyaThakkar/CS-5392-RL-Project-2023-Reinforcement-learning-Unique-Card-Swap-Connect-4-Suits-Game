from game import game
import gymnasium as gym
import random as rd
from gymnasium import Env, spaces 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
    env = game()
    for i in range(1):
        #playRandomVsRandomPlapyerTest1(env, 9999, False)
        playWithAgentModel(env,'C:/w/ttu/RL/model_8000_20230211201405.h5');

def playRandomVsRandomPlapyer(env, total, logLevelHigh):
    match = 0
    for i in range(total):
        match = match + 1
        env.reset()
        Done = False
        while Done == False:
            if(logLevelHigh):
                env.render()
            state, reward, Done, info = env.randoVsRando()
            if Done:
                env.render()
                print('reward = ',reward, "match = ",match)

# Check how many turns does it on average take to complete this game
# using random players
def playRandomVsRandomPlapyerTest1(env, total, logLevelHigh):
    match = 0
    completed = 0 
    draw = 0
    for i in range(total):
        match = match + 1
        env.reset()
        Done = False
        while Done == False:
            if(logLevelHigh):
                env.render()
            state, reward, Done, info = env.randoVsRando()
            if Done:
                #env.render()
                if reward == 0:
                    draw += 1
                elif reward == 100 or reward == -100:
                    completed += 1
                else:
                    print('Weird, why am I here ?')
    print('completed = ', completed , ' draw = ' ,draw , \
        ' Percentage Completed = ',(completed / (completed + draw)) )

def playWithAgentModel(env, file):
    model = create_q_model((18,),46)
    model.load_weights(file)
    state  = env.reset()
    while True:
        env.render()
        print()
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = model(state_tensor, training=False)
        # Take best action
        suggestedAction = tf.argmax(action_probs[0]).numpy()
        print('suggested action is ', suggestedAction)
        print('Enter your action')
        input_action = input()
        state, reward, Done, info = env.step(int(input_action))
        print('reward = ',reward)
        if Done:
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