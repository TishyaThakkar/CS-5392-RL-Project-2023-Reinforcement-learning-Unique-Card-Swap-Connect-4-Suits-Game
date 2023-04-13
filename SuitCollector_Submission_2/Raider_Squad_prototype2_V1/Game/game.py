import random as rd;

import numpy
import numpy as np;
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os.path

import phase1
from phase1 import *


# Suite Collector Game Class File
class game():

    def __init__(self):
        # Setting up the Game
        # Pieces - Format AB, A-> Suite , B-> Number
        self.pieces = np.array([1,2,3,4,0,0,0,0,0,0,0,0,-1,-2,-3,-4])
        self.board = np.copy(self.pieces)

        #populating actions
        # swap x and y  
        # actions[i] = [x,y]
        self.actions = np.full([16*8,2], -1)
        #actionsXY[x,y] = i , reverse Map of above map
        self.actionsXY = np.full([16,16],-1)
        self.actionsCount = 0
        # The suites in the Game.
        self.suites = np.array([1,-1])
        
        # For each cell populating the valid actions.
        for i in range(0,4):
            for j in range(0,4):
                #8 things
                a,b = i,j

                # Top Left
                c,d = i-1,j-1
                self.processAction(a,b,c,d)

                # Top
                c,d = i,j-1
                self.processAction(a,b,c,d)

                # Top Right
                c,d = i+1,j-1
                self.processAction(a,b,c,d)

                # Right
                c,d = i+1,j
                self.processAction(a,b,c,d)

                # Bottom Right
                c,d = i+1,j+1
                self.processAction(a,b,c,d)

                # Bottom 
                c,d = i,j+1
                self.processAction(a,b,c,d)

                # Bottom Left 
                c,d = i-1,j+1
                self.processAction(a,b,c,d)

                # Left 
                c,d = i-1,j-1
                self.processAction(a,b,c,d)
        
        self.models = [self.create_q_model_iron((16,),42), self.create_q_model_gold((16,),42), self.create_q_model_diamond1((16,),42), self.create_q_model_diamond2((16,),42)]
        
    def convertToPhase2Board(self, board, suite):
        #Converts a board to phase2 Board
        userSuit = suite[0]
        agentSuit = suite[1]
        self.board = np.full([16], 0)
        for i in range(16):
            card = board[i]
            suit = int(card/10)
            value = int(card%10)
            if suit == userSuit:
                self.board[i] = -value
            if suit == agentSuit:
                self.board[i] = value

    
    def trackAces(self):
        # Track Aces in the board, Format - A1 | A = {1,2,3,4}
        self.aces = np.full([3],-1)
        for i in range(0,len(self.board)):
            x = self.board[i] 
            if self.board[i] == 1 or self.board[i] == -1:
                self.aces[self.board[i]] = i
    
   
    # Checks if a co-ordinate is valid or not
    def isValid(self, x):
        if x >=0 and x < 4:
            return True
        return False

    def normalizeBoard(self):
        return (self.board / 4)

    def normalizeBoardForAssistanceModel(self):
        return (self.board / -4)

    # Populate both action Maps with an action.
    def populateAction(self,x,y):
        self.actionsXY[x,y] = self.actionsCount
        self.actions[self.actionsCount, 0] = x
        self.actions[self.actionsCount, 1] = y
        self.actionsCount += 1

    # Processes a action
    #   checks if the destination coordinates are correct.
    #   checks if the action already exists. swap(X,Y) = swap(Y,X)
    def processAction(self,a,b,c,d):
        if self.isValid(c) and self.isValid(d):
            x, y = 4 * a + b, 4 * c + d
            if( x > y):
                x,y = y,x
            if(self.actionsXY[x,y] == -1):
                self.populateAction(x,y)
    
    #Random players
    def agentRock(self, state_tensor, training=False):
        #Computer makes your random valid move.
        # repeat - pick a random move and check until its valid
        action = 0
        card1Pos = None
        card2Pos = None
        while True:
            action = rd.randint(0,41)
            card1Pos = self.actions[action][0]
            card2Pos = self.actions[action][1]

            card1 = self.board[card1Pos]
            card2 = self.board[card2Pos]

            if ( card1 < 0 or card2 < 0 ):
                continue
            else:
                break

        return_obj = [0] * 41
        return_obj[action] = 1.0
        return tf.convert_to_tensor([return_obj])

    def randoVsRando(self):
        #Computer makes your random valid move.
        # repeat - pick a random move and check until its valid
        action = 0
        card1Pos = None
        card2Pos = None
        while True:
            action = rd.randint(0,41)
            card1Pos = self.actions[action][0]
            card2Pos = self.actions[action][1]

            card1 = self.board[card1Pos]
            card2 = self.board[card2Pos]

            if (
                card1 < 0 or \
                card2 < 0
                ):
                continue
            else:
                break

        board, reward, done, info = self.step(action)
        return board, reward, done, info, action
    
        
    # Check if a Suite won the game or not.
    def checkIfSuiteWon(self, suite):
        # get position of ace in suite
        acePos = self.aces[suite]
        x = [0,-1,-2,-3]
        if(suite == 1):
            x = [0,1,2,3]
        
        # Diagonals
        if acePos == 0:
            if self.board[5]  == self.board[0]+x[1] and \
               self.board[10] == self.board[0]+x[2] and \
               self.board[15] == self.board[0]+x[3] :
                return True
        if acePos == 3:
            if self.board[6]  == self.board[3]+x[1] and \
               self.board[9]  == self.board[3]+x[2] and \
               self.board[12] == self.board[3]+x[3] :
                return True
        if acePos == 12:
            if self.board[9]  == self.board[12]+x[1] and \
               self.board[6]  == self.board[12]+x[2] and \
               self.board[3] == self.board[12]+x[3] :
                return True
        if acePos == 15:
            if self.board[10]  == self.board[15]+x[1] and \
               self.board[5]  == self.board[15]+x[2] and \
               self.board[0] == self.board[15]+x[3] :
                return True

        acePosX , acePosY = int(acePos % 4) , int(acePos / 4) 
        # Top Row
        if acePosY == 0:
            if self.board[acePos + 4] == self.board[acePos] + x[1] and \
               self.board[acePos + 8] == self.board[acePos] + x[2] and \
               self.board[acePos + 12] == self.board[acePos] + x[3]  :
                return True

        # Bottom Row
        if acePosY == 3:
            if self.board[acePos - 4] == self.board[acePos] + x[1] and \
               self.board[acePos - 8] == self.board[acePos] + x[2] and \
               self.board[acePos - 12] == self.board[acePos] + x[3]  :
                return True

        # Left Column
        if acePosX == 0:
            if self.board[acePos + 1] == self.board[acePos] + x[1] and \
               self.board[acePos + 2] == self.board[acePos] + x[2] and \
               self.board[acePos + 3] == self.board[acePos] + x[3]  :
                return True

        # Right Column
        if acePosX == 3:
            if self.board[acePos - 1] == self.board[acePos] + x[1] and \
               self.board[acePos - 2] == self.board[acePos] + x[2] and \
               self.board[acePos - 3] == self.board[acePos] + x[3]  :
                return True

        return False
        
    
    # Renders the game on to the console.
    def render(self):
        print()
        for i in range(0,4):
            for j in range(0,4):
                print(self.board[(i*4)+j], ",  ", end="")
            print()
        print("\nSuite: YOU =" , 1, " , ME = ", -1 )
        print("aces position = ", self.aces)
        print("board = " , self.board)
        print("normalize board = ", self.normalizeBoard())
        print("time = ", self.time)
    

    # Networks
    def create_q_model_iron(self, state_shape, total_actions):
        """ Create a Q model for Agent Iron"""
        # input layer
        inputs = layers.Input(shape=state_shape)

        # Hidden layers
        layer1 = layers.Dense(40, activation="relu")(inputs)
        layer2 = layers.Dense(40, activation="relu")(layer1)

        # output layer    
        action = layers.Dense(total_actions, activation="linear")(layer2)

        model = keras.Model(inputs=inputs, outputs=action)
        model.load_weights('./models/model-iron.h5')
        return model
    
    def create_q_model_gold(self, state_shape, total_actions):
        """ Create a Q model for Agent Gold"""
        # input layer
        inputs = layers.Input(shape=state_shape)

        # Hidden layers
        layer1 = layers.Dense(500, activation="relu")(inputs)
        layer2 = layers.Dense(250, activation="relu")(layer1)

        # output layer    
        action = layers.Dense(total_actions, activation="linear")(layer2)

        model = keras.Model(inputs=inputs, outputs=action)
        model.load_weights('./models/model-gold.h5')
        return model

    def create_q_model_diamond1(self, state_shape, total_actions):
        """ Create a Q model for Agent Diamond 1"""
        # input layer
        inputs = layers.Input(shape=state_shape)

        # Hidden layers
        layer1 = layers.Dense(500, activation="relu")(inputs)
        layer2 = layers.Dense(250, activation="relu")(layer1)

        # output layer    
        action = layers.Dense(total_actions, activation="linear")(layer2)

        model = keras.Model(inputs=inputs, outputs=action)
        model.load_weights('./models/model-gold.h5')
        return model


    def create_q_model_diamond2(self, state_shape, total_actions):
        """ Create a Q model for Agent Diamond 2"""
        # input layer
        inputs = layers.Input(shape=state_shape)
        layer0 = layers.Reshape((-1,4, 4,1), input_shape=state_shape)(inputs)

        # 128 filters , 2x2 size, Deep Convoluted Layer with 3 hidden layers
        layer1 =  layers.Conv2D(128, 2, strides=1, activation="relu")(layer0)
        layer2 =  layers.Conv2D(128, 2, strides=1, activation="relu")(layer1)
        layer3 =  layers.Conv2D(128, 2, strides=1, activation="relu")(layer2)
        layer31 = layers.Flatten()(layer3)

        # A hidden Layer
        layer4 = layers.Dense(77, activation="relu")(layer31)

        # output layer    
        action = layers.Dense(total_actions, activation="linear")(layer4)
        
        model = keras.Model(inputs=inputs, outputs=action)
        model.load_weights('./models/model-diamond2.h5')
        return model
    
    def process_request(self, message):
        print(message)
        if message['suits'][0] == 0 or message['suits'][1] == 0:
            return self.processSuits(message)
        else:
            return self.processPlay(message)

    def processSuits(self, message):
        userSuit = message['suits'][0]
        agentSuit = message['suits'][1]
        agent_id = message['playAgainst']
        data = {}
        data['board'] = None
        data['processSuits'] = True

        # Agent already picked a Suit, 
        if agentSuit != 0:
            data['msg_e'] = 'Please pick your suit.'
            return data
        
        # User is playing against Random Agent
        if agent_id == 0:
            agentSuit = random.randint(1, 4)
            while agentSuit == userSuit:
                agentSuit = random.randint(1, 4)

            data['suit'] = str(agentSuit)
            return data
        else:
        # We have two Algorithms to choose from:
            # 1. Q Learning
            # 2. Using Phase 2 Models 
                
            # Code for Q-Learning ... 
            """
            if [UI Option is Phase1 Model]:
                # Some Code
                suits = [1, 2, 3, 4]
                suits_to_names = ['Clubs', 'Hearts', 'Spades', 'Diamonds']
                answer = phase1.main(True, message['board'])
                if answer == 0:
                    data['suit'] = '3'
                elif answer == 1:
                    data['suit'] = '2'
                elif answer == 2:
                    data['suit'] = '1'
                else:
                    data['suit'] = '4'
                return data
            else:
                
            """
            # Code using Phase 2 Models
            
            # Suits for User.
            userValidSuits = [1,2,3,4]
            if userSuit != 0:
                    userValidSuits = [userSuit]
            
            agentValidSuits = []
            for i in range(1,5):
                if userSuit != i:
                    agentValidSuits.append(i)
            
            best_suit = -100000
            best_QValue = -1000000

            for i in agentValidSuits:
                    rewards = []
                    for j in userValidSuits:
                        if i!=j:
                            self.convertToPhase2Board(message['board'], [j, i])
                            rewards.append(self.getMaxQValues(agent_id-1))
                    # Since user will pick the best option for him always
                    curernt_best_reward = min(rewards)
                    if(curernt_best_reward > best_QValue):
                        best_suit = i
                        best_QValue = curernt_best_reward;
            
            data['suit'] = str(best_suit) 
            return data



    def processPlay(self, message):
        data = {};
        data['board'] = None
        data['processPlay'] = True
        self.convertToPhase2Board(message['board'], message['suits']);
        self.trackAces()
        if self.checkIfSuiteWon(1) or self.checkIfSuiteWon(-1) :
            data['msg_i'] = 'Game Has Ended Already';
            return data;
    
        if(len(message['selectedCards']) == 2):
            # check if the selected cards make a valid aciton or not.
            x = message['selectedCards'][0]
            y = message['selectedCards'][1]
            if(self.checkUserValidAction(x,y)):
                # perform the move and track aces
                self.board[x] , self.board[y] = self.board[y] , self.board[x]
                message['board'][x] , message['board'][y] = message['board'][y] , message['board'][x]
                if self.board[x] == -1:
                    self.aces[-1] = x
                if self.board[y] == -1:
                    self.aces[-1] = y

                # check if User won ?
                if self.checkIfSuiteWon(-1):
                    data['msg_s'] = 'You Won!!!'
                    data['board'] = message['board']
                    return data;
            else:
                data['msg_e'] = 'Invalid Action';
                return data;
    
        # your move next!!
        # pick an action using the agent. 
        agent_id = message['playAgainst']
        action = -1; x = -1; y = -1; isRandom = True
        if(agent_id == 0):
            #use random agent to pick an action.
            action,x,y,isRandom = self.getValidRandomAction()
        else:
            action,x,y,isRandom = self.getAgentAction(agent_id-1)

        data['isRandomAction'] = isRandom
        print(action,x,y, isRandom)

        #perform action and track aces
        self.board[x] , self.board[y] = self.board[y] , self.board[x]
        message['board'][x] , message['board'][y] = message['board'][y] , message['board'][x]
        if self.board[x] == 1:
            self.aces[1] = x
        if self.board[y] == 1:
            self.aces[1] = y

        data['board'] =  message['board']
        data['aiSelectedCards'] = [x,y]
        
        if self.checkIfSuiteWon(1):
            data['msg_e'] = 'You Lost!!!'
            return data;
    
        return data

    def checkUserValidAction(self, x, y):
        if( x > y):
            x,y = y,x
        if(self.actionsXY[x,y] == -1):
            return False
        
        card1 = self.board[x]
        card2 = self.board[y]

        if ( card1 > 0 or card2 > 0 ):
            return False
        else:
            return True
        
    def getValidRandomAction(self):
        #Computer makes your random valid move.
        # repeat - pick a random move and check until its valid
        action = 0
        card1Pos = None
        card2Pos = None
        while True:
            action = rd.randint(0,41)
            card1Pos = self.actions[action][0]
            card2Pos = self.actions[action][1]

            card1 = self.board[card1Pos]
            card2 = self.board[card2Pos]

            if ( card1 < 0 or card2 < 0 or (card1 ==0 and card2 ==0)):
                continue
            else:
                break
        return action, card1Pos, card2Pos, True

    def getAgentAction(self, agent_id):
        state_tensor = tf.convert_to_tensor(self.normalizeBoard())
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_qvalues = self.models[agent_id](state_tensor, training=False)
        print("action_qvalues: ", action_qvalues)
        # Take best action
        action = tf.argmax(action_qvalues[0]).numpy() 
        card1Pos = self.actions[action][0]
        card2Pos = self.actions[action][1]
        card1 = self.board[card1Pos]
        card2 = self.board[card2Pos]
        if(card1< 0 or card2 <0 ):
            action , x, y = self.performValidRandomAction()
            return action , x, y , True
        return action, card1Pos, card2Pos, False

    def getMaxQValues(self, agent_id):
        state_tensor = tf.convert_to_tensor(self.normalizeBoard())
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_qvalues = self.models[agent_id](state_tensor, training=False)
        return max(action_qvalues[0].numpy())




def main():
    return 

if __name__ == "__main__":
    main()

###
#   Position:
#   0  1  2  3 
#   4  5  6  7
#   8  9  10 11
#   12 13 14 15

#   action [X,Y] // Swap position X and Y, see above
#   0 [0 4]	        #   21 [ 6 10]		#   41 [14 15]
#   1 [0 5]		    #   22 [ 6 11]		#   42 Pick Suite 1
#   2 [0 1]		    #   23 [6 7]		#   43 Pick Suite 2
#   3 [1 4]		    #   24 [ 7 10]		#   44 Pick Suite 3
#   4 [1 5]		    #   25 [ 7 11]		#   45 Pick Suite 4
#   5 [1 6]		    #   26 [ 8 12]		
#   6 [1 2]		    #   27 [ 8 13]		
#   7 [2 5]		    #   28 [8 9]		
#   8 [2 6]		    #   29 [ 9 12]		
#   9 [2 7]		    #   30 [ 9 13]		
#   10 [2 3]		#   31 [ 9 14]		
#   11 [3 6]		#   32 [ 9 10]		
#   12 [3 7]		#   33 [10 13]		
#   13 [4 8]		#   34 [10 14]		
#   14 [4 9]		#   35 [10 15]		
#   15 [4 5]		#   36 [10 11]		
#   16 [5 8]		#   37 [11 14]		
#   17 [5 9]		#   38 [11 15]		
#   18 [ 5 10]		#   39 [12 13]		
#   19 [5 6]		#   40 [13 14]		
#   20 [6 9]				

###
