from game_ma import game
import gymnasium as gym
import random as rd
from gymnasium import Env, spaces 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
    env = game('./model.h5')
    for i in range(1):
        #playRandomVsRandomPlapyerTest1(env, 9999, False)
        #playWithAgentModel(env,'C:/w/ttu/RL/model_80000_20230214214113.h5');
        #playAMinGame()
        #playWithAgentModelMini(game(), './assistance_model/model_16200_20230221100854.h5' , 3)
        playwithAssistedAgent(env,20)
        #randomProbabilityChecker(99999)
        #testPlayAMinGame()
        pass

def testPlayAMinGame():
    env = gameMin()
    env.TestcheckIfSuiteWon([-2,0,0,-1,   1,2,3,4,  -4,0,0,0,  0,0,0,-4  ], [-1,4,3] , 1)

def playwithAssistedAgent(env, total):
    for i in range(total):
        env.reset()
        done = False
        while done != True:
            env.render()
            print('enter your action')
            x = int(input())
            board , reward , done, info = env.step(x)
            print("reward = ", reward)
            print("info = ", info)



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

# def playWithAgentModel(env, file):
#     model = create_q_model((18,),46)
#     model.load_weights(file)
#     state  = env.reset()
#     while True:
#         env.render()
#         print()
#         state_tensor = tf.convert_to_tensor(state)
#         state_tensor = tf.expand_dims(state_tensor, 0)
#         action_probs = model(state_tensor, training=False)
#         # Take best action
#         suggestedAction = tf.argmax(action_probs[0]).numpy()
#         print('suggested action is ', suggestedAction)
#         print('Enter your action')
#         input_action = input()
#         state, reward, Done, info = env.step(int(input_action))
#         print('reward = ',reward)
#         if Done:
#             break

# def create_q_model(state_shape, total_actions):
#     # input layer
#     inputs = layers.Input(shape=state_shape)

#     # Hidden layers
#     layer1 = layers.Dense(18, activation="relu")(inputs)
#     layer2 = layers.Dense(64, activation="relu")(layer1)
#     layer3 = layers.Dense(90, activation="relu")(layer2)
#     layer4 = layers.Dense(64, activation="relu")(layer3)   
#     #layer5 = layers.Dense(10, activation="relu")(layer4)

#     # output layer    
#     action = layers.Dense(total_actions, activation="linear")(layer4)

#     return keras.Model(inputs=inputs, outputs=action)

def playAMinGame():
    env = gameMin()
    env.reset()
    done = False
    while done != True:
        env.render()
        print('enter your action')
        x = int(input())
        board , reward , done, info = env.step(x)
        print("reward = ", reward)

def playWithAgentModelMini(env, file, repeat):
    model = create_q_model((16,),42)
    model.load_weights(file)
    state  = env.reset()
    for i in range(repeat):
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
    layer1 = layers.Dense(1000, activation="relu")(inputs)
    #layer2 = layers.Dense(40, activation="relu")(layer1)
    #layer3 = layers.Dense(36, activation="relu")(layer2)
    #layer4 = layers.Dense(54, activation="relu")(layer3)   
    #layer5 = layers.Dense(10, activation="relu")(layer4)

    # output layer    
    action = layers.Dense(total_actions, activation="linear")(layer1)

    return keras.Model(inputs=inputs, outputs=action)


#check correct move probability
def randomProbabilityChecker(t):
    env = game()
    c = 0
    f = 0
    ta = 0
    for i in range(t):
        env.reset()
        done = False
        while done == False:
            action = rd.randint(0,41)
            board, reward, done, info = env.step(action)
            if reward == 1 or reward == 0:
                c+=1
            else:
                f += 1
            ta += 1
    template = "totl actions = {} , random good moves probability = {}, bad moves probability = {}"
    print(template.format((c+f), (c*100/(c+f)), (f*100/(c+f))))
    print(c,f,ta)



if __name__=="__main__":
    main()

