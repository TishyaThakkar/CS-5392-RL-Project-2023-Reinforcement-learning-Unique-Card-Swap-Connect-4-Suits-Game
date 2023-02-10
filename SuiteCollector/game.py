import random as rd;
import numpy as np;
import gymnasium as gym;

from gymnasium import Env, spaces 

class game(gym.Env):

    def __init__(self):
        # Setting up the Game
        
        # Pieces - Format AB, A-> Suite , B-> Number
        self.pieces = np.array([11,12,13,14,21,22,23,24,31, \
                                32,33,34,41,42,43,44])
        self.board = np.copy(self.pieces)
        #populating actions
        # swap x and y  
        # actions[i] = [x,y]
        self.actions = np.full([16*8,2], -1)
        #actionsXY[x,y] = i
        self.actionsXY = np.full([16,16],-1)
        self.actionsCount = 0
        self.suites = np.array([1,2,3,4])

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

        
        # special actions [42=1, 43=2, 44=3, 45=4]
        self.specialActionOptions = np.array([1,2,3,4])
        # Zero is computer goes first
        # 1 is Agent goes first
        self.turn = -1
        self.startGame()

        

        
    def startGame(self):
        # Initialize a random board
        self.board = np.copy(self.pieces)
        np.random.shuffle(self.board)
        self.board = np.append(self.board, [-1,-1])

        # turn , 0 => Computer , 1 => Player/Agent 
        self.turn = int(rd.getrandbits(1))

        # Track Aces
        self.aces = np.full([5],-1)
        for i in range(0,len(self.board)):
            if int(self.board[i] % 10) == 1:
                self.aces[int(self.board[i] / 10)] = i
        #print(self.aces)

        # first play pick a suite
        if self.turn == 0:
            self.board[-1] = np.random.choice(self.suites, None)
            self.turn = 1

        #print(self.actionsCount)
        #print(self.actions)
        #for i in range(44):
            #print(i , self.actions[i])
        #print(self.actionsXY)
   

    def isValid(self, x):
        if x >=0 and x < 4:
            return True
        return False

    def populateAction(self,x,y):
        self.actionsXY[x,y] = self.actionsCount
        self.actions[self.actionsCount, 0] = x
        self.actions[self.actionsCount, 1] = y
        self.actionsCount += 1

    def processAction(self,a,b,c,d):
        if self.isValid(c) and self.isValid(d):
            x, y = 4 * a + b, 4 * c + d
            if( x > y):
                x,y = y,x
            if(self.actionsXY[x,y] == -1):
                self.populateAction(x,y)
        
    def reset(self):
        self.board = np.copy(self.pieces)
        np.random.shuffle(self.board)
        self.board = self.board.reshape([4,4])
        self.Players = [-1,-1]
        self.turn = 0
        self.gameStarted = False

    def step(self, action):
        # Player plays a move 
        if(self.board[-2] == -1 and action < 42):
            return self.board, -100, True, None
        elif(self.board[-2] == -1):
            choosenSuite = (action-42+1)
            if(choosenSuite == self.board[-1]):
                print('bad suite')
                return self.board, -100, True, None
            self.board[-2] = choosenSuite
            print('You picked the Suite: ', choosenSuite)
        elif(action >=0 and action <  42):
            #player makes an action

            #check if the action is valid or not
            # an action is valid if it does not swap apponents cards.
            card1Pos = self.actions[action][0]
            card2Pos = self.actions[action][1]

            card1 = self.board[card1Pos]
            card2 = self.board[card2Pos]

            opponentSuite = self.board[-1]

            if (
                int(card1 / 10) == opponentSuite or \
                int(card2 / 10) == opponentSuite
                ):
                return self.board, -100, True, None

            # else action is valid
            # perform the action if valid
            self.board[card1Pos] = card2
            self.board[card2Pos] = card1

            # Track aces if required.
            if int(self.board[card1Pos] % 10) == 1:
                self.aces[int(self.board[card1Pos] / 10)] = card1Pos

            if int(self.board[card2Pos] % 10) == 1:
                self.aces[int(self.board[card2Pos] / 10)] = card2Pos

            # Check if you have won.
            # set reward that we need to return.
            won = self.checkIfSuiteWon(self.board[-2])
            if won:
                return self.board, 100, True, None
        else:
            print("Unkown Error")
            return self.board, -100, True, None

        print('You made a move ', action)
        
        # Computer makes a move.
        if(self.board[-1] == -1):
            takenSuite = self.board[-2]
            mySuite = np.random.choice(self.suites, None)
            while(mySuite == takenSuite):
                mySuite = np.random.choice(self.suites, None)
            self.board[-1] = mySuite
            print('I picked a Suite ', mySuite, )
        else:
            #Computer makes a random valid move.
            # repeat - pick a random move and check until its valid

            myaction = 0
            card1Pos = None
            card2Pos = None
            while True:
                myaction = rd.randint(0,41)
                card1Pos = self.actions[myaction][0]
                card2Pos = self.actions[myaction][1]

                card1 = self.board[card1Pos]
                card2 = self.board[card2Pos]

                opponentSuite = self.board[-2]

                if (
                    int(card1 / 10) == opponentSuite or \
                    int(card2 / 10) == opponentSuite
                    ):
                    continue
                else:
                    break
            
            # Perform the action if valid
            self.board[card1Pos] = card2
            self.board[card2Pos] = card1
            print("Computer playing action ",myaction)
            # Track aces if required.
            if int(self.board[card1Pos] % 10) == 1:
                self.aces[int(self.board[card1Pos] / 10)] = card1Pos

            if int(self.board[card2Pos] % 10) == 1:
                self.aces[int(self.board[card2Pos] / 10)] = card2Pos

            # Check if the computer won.
            # set reward accordingly and return.
            won = self.checkIfSuiteWon(self.board[-1])
            if won:
                return self.board, -100, True, None

            

        
        # finally
        self.render()
        return self.board, 0, False, None 

    def checkIfSuiteWon(self, suite):
        # get position of ace in suite
        acePos = self.aces[suite]
        # Diagonals
        if acePos == 0:
            if self.board[5]  == self.board[0]+1 and \
               self.board[10] == self.board[0]+2 and \
               self.board[15] == self.board[0]+3 :
                return True
        if acePos == 3:
            if self.board[6]  == self.board[3]+1 and \
               self.board[9]  == self.board[3]+2 and \
               self.board[12] == self.board[3]+3 :
                return True
        if acePos == 12:
            #print('I herer 12')
            if self.board[9]  == self.board[12]+1 and \
               self.board[6]  == self.board[12]+2 and \
               self.board[3] == self.board[12]+3 :
                return True
        if acePos == 15:
            if self.board[10]  == self.board[15]+1 and \
               self.board[5]  == self.board[15]+2 and \
               self.board[0] == self.board[15]+3 :
                return True

        acePosX , acePosY = int(acePos % 4) , int(acePos / 4) 
        # Top Row
        if acePosY == 0:
            if self.board[acePos + 4] == self.board[acePos] + 1 and \
               self.board[acePos + 8] == self.board[acePos] + 2 and \
               self.board[acePos + 12] == self.board[acePos] + 3  :
                return True

        # Bottom Row
        if acePosY == 3:
            if self.board[acePos - 4] == self.board[acePos] + 1 and \
               self.board[acePos - 8] == self.board[acePos] + 2 and \
               self.board[acePos - 12] == self.board[acePos] + 3  :
                return True

        # Left Column
        if acePosX == 0:
            if self.board[acePos + 1] == self.board[acePos] + 1 and \
               self.board[acePos + 2] == self.board[acePos] + 2 and \
               self.board[acePos + 3] == self.board[acePos] + 3  :
                return True

        # Right Column
        if acePosX == 3:
            if self.board[acePos - 1] == self.board[acePos] + 1 and \
               self.board[acePos - 2] == self.board[acePos] + 2 and \
               self.board[acePos - 3] == self.board[acePos] + 3  :
                return True

        return False
        
        

    def TestcheckIfSuiteWon(self, board, aces, suite):
        self.board = board
        self.aces = aces
        print(self.checkIfSuiteWon(suite))
    
    def render(self):
        print()
        for i in range(0,4):
            for j in range(0,4):
                print(self.board[(i*4)+j], " ", end="")
            print()
        print("\nSuite: YOU =" , self.board[-2], " , ME = ", self.board[-1] )
        print("aces position = ", self.aces)
        print("board = " , self.board)


        

c = game()

'''
c.TestcheckIfSuiteWon(np.array([ 31, 32, 33, 34, \
                                 11, 12, 13, 14, \
                                 21, 22, 23, 24, \
                                 41, 43, 42, 44  \
                                 ]) , np.array([-1, 4, 8, 0, 12]), 4)

c.TestcheckIfSuiteWon(np.array([ 14, 41, 21, 31, \
                                 13, 42, 22, 32, \
                                 12, 43, 23, 33, \
                                 11, 44, 24, 34  \
                                 ]) , np.array([-1, 12, 2, 3, 1]), 4)

c.TestcheckIfSuiteWon(np.array([ 31, 41, 21, 14, \
                                 22, 42, 13, 32, \
                                 43, 12, 23, 33, \
                                 11, 44, 24, 34  \
                                 ]) , np.array([-1, 12, 2, 0, 1]), 1)
'''


c.render()
while True:
    print()
    print('Enter your action')
    input_action = input()
    state, reward, Done, inof = c.step(int(input_action))
    print('reward = ',reward)
    if Done:
        break

comments = '''

Position:
0  1  2  3 
4  5  6  7
8  9  10 11
12 13 14 15

actoin [X,Y] // Swap position X and Y, see above
0 [0 4]
1 [0 5]
2 [0 1]
3 [1 4]
4 [1 5]
5 [1 6]
6 [1 2]
7 [2 5]
8 [2 6]
9 [2 7]
10 [2 3]
11 [3 6]
12 [3 7]
13 [4 8]
14 [4 9]
15 [4 5]
16 [5 8]
17 [5 9]
18 [ 5 10]
19 [5 6]
20 [6 9]
21 [ 6 10]
22 [ 6 11]
23 [6 7]
24 [ 7 10]
25 [ 7 11]
26 [ 8 12]
27 [ 8 13]
28 [8 9]
29 [ 9 12]
30 [ 9 13]
31 [ 9 14]
32 [ 9 10]
33 [10 13]
34 [10 14]
35 [10 15]
36 [10 11]
37 [11 14]
38 [11 15]
39 [12 13]
40 [13 14]
41 [14 15]
42 Pick Suite 1
43 Pick Suite 2
44 Pick Suite 3
45 Pick Suite 4

'''
