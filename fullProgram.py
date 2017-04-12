import numpy as np
import time  # Is the feedback to the neural network appropriate?
import random
import re
import copy

np.set_printoptions(suppress=True)
from random import shuffle
populationStats = []
printMode = False

split = False
double = False

splitTrain = False
doubleTrain = False
drawFoldTrain = False

populationWeights = [0,0,0]
populationWinRate = [0,0,0]

overallWin = 0
overallWeights = []
currentWin = 0
previousWin = 0
previousWeights = []

#thorpes strategy charts
thorpesSimple = ["H", "H", "D", "D", "H", "S", "S", "S",
                 "S", "S", "H", "D", "D", "D", "H", "S",
                 "S", "S", "S", "S", "H", "D", "D", "D",
                 "S", "S", "S", "S", "S", "S", "H", "D", "D",
                 "D", "S", "S", "S", "S", "S", "S", "H",
                 "D", "D", "D", "S", "S", "S", "S", "S", "S",
                 "H", "H", "D", "D", "H", "H", "H", "H", "H",
                 "S", "H", "H", "D", "D", "H", "H", "H", "H",
                 "H", "S", "H", "H", "D", "D", "H", "H", "H",
                 "H", "H", "S", "H", "H", "H", "D", "H", "H",
                 "H", "H", "H", "S", "H", "H", "H", "H", "H",
                 "H", "H", "H", "H", "S"]

thorpesAce = ["H", "H", "H", "H", "H", "S", "S", "S",
              "H", "H", "H", "H", "D", "Ds", "S", "S",
              "H", "H", "D", "D", "D", "Ds", "S", "S",
              "D", "D", "D", "D", "D", "Ds", "S", "S",
              "D", "D", "D", "D", "D", "Ds", "S", "S",
              "H", "H", "H", "H", "H", "S", "S", "S",
              "H", "H", "H", "H", "H", "S", "S", "S",
              "H", "H", "H", "H", "H", "H", "S", "S",
              "H", "H", "H", "H", "H", "H", "S", "S",
              "H", "H", "H", "H", "H", "H", "S", "S"]

thorpesPair = ["P", "P", "H", "D", "P", "P", "P", "P","S", "P",
               "P", "P", "H", "D", "P", "P","P", "P", "S", "P",
               "P", "P", "H", "D","P", "P", "P", "P", "S", "P",
               "P", "P","P", "D", "P", "P", "P", "P", "S", "P",
               "P", "P", "P", "D", "P", "P", "P", "P","S", "P",
               "P", "P", "H", "D", "H", "P","P", "S", "S", "P",
               "H", "H", "H", "D","H", "H", "P", "P", "S", "P",
               "H", "H","H", "D", "H", "H", "P", "P", "S", "P",
               "H", "H", "H", "H", "H", "H", "P", "S","S", "P",
               "H", "H", "H", "H", "H", "H","P", "S", "S", "P"]

# enumerate isn't required, just loop for each item in the list

def rando(bestWeights): #create weights based on best agents of previous populations
    weights = copy.deepcopy(bestWeights)
    for current in range(len(weights) - 1, -1, -1):
        for inNode in range(len(weights[current])):
            if random.randint(0, 9) >= 5:
                if len(weights[current][inNode]) <= 1:
                    rand = random.uniform(0, 1)
                    weights[current][inNode][0] = rand
                else:
                    for outNode in range(len(weights[current][inNode])):
                        rand = random.uniform(0, 0.1)
                        weights[current][inNode][outNode] = rand
    return weights

class NeuralNetwork:  # Need to sort out where the class parameters should go
    def __init__(self, structure, startWeights):  # rename hiddenNodeOutput etc #IS THE SELF. ACTUALLY REQUIRED?
        # network structure
        self.structure = structure  # Each number represents the weights in each layer of the NN
        self.weights = []
        self.nodeOutputs = []  # Outputs from each node in each layer
        self.nodeInputs = []
        self.deltas = []

        if startWeights == 0: #if no weights then create random weights
            for i in range(len(self.structure) - 1):
                self.weights.append(
                    np.random.rand(self.structure[i], self.structure[i + 1]))  # Create and initialise weights
        else:
            self.weights = startWeights

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def propagate(self, testSet, nnType):
        self.nodeInputs = []
        self.nodeOutputs = []
        inputs = []
        if nnType == 0:
            for i in range(4, 21):
                if i == testSet[0]:
                    inputs.append(1)
                else:
                    inputs.append(0)

            for i in range(2, 12):
                if i == testSet[1]:
                    inputs.append(1)
                else:
                    inputs.append(0)
            inputs.append(testSet[2])
        elif nnType == 1:
            for i in range(2, 12):
                if i == testSet[0]:
                    inputs.append(1)
                else:
                    inputs.append(0)
            for i in range(2, 12):
                if i == testSet[1]:
                    inputs.append(1)
                else:
                    inputs.append(0)
        else:
            for i in range(4, 21):
                if i == testSet[0]:
                    inputs.append(1)
                else:
                    inputs.append(0)

            for i in range(2, 12):
                if i == testSet[1]:
                    inputs.append(1)
                else:
                    inputs.append(0)
            inputs.append(testSet[2])

        for i in range(len(self.structure)):
            if i == 0:
                self.nodeInputs.append(inputs)
                self.nodeOutputs.append(self.nodeInputs[-1])
            elif i < len(self.structure) - 1:
                self.nodeInputs.append(np.dot(self.nodeOutputs[-1], self.weights[i - 1]))
                self.nodeOutputs.append(self.sigmoid(self.nodeInputs[-1]))
            else:
                self.nodeInputs.append(np.dot(self.nodeOutputs[-1], self.weights[i - 1]))
                self.nodeOutputs.append(self.sigmoid(self.nodeInputs[-1]))
        return (self.nodeOutputs[-1])

    def BP(self, expected):
        self.deltas = []
        lastLayer = []
        for current in range(len(self.weights) - 1, -1, -1):
            if current == len(self.weights) - 1:
                for inNode in range(len(self.weights[-1])):
                    if len(self.weights[-1][inNode]) <= 1:
                        lastLayer.append([(self.nodeOutputs[-1][0] - expected[0]) * (
                            self.nodeOutputs[-1][0] * (1 - self.nodeOutputs[-1][0])) * self.nodeOutputs[-2][inNode]])
                    else:
                        for outNode in range(len(self.weights[-1][inNode])):
                            lastLayer.append([(self.nodeOutputs[-1][outNode] - expected[outNode]) * (
                                self.nodeOutputs[-1][outNode] * (1 - self.nodeOutputs[-1][outNode])) *
                                              self.nodeOutputs[-2][inNode]])
                self.deltas.append(lastLayer)
                layerDeltas = []
            elif current == len(self.weights) - 2:
                layer = current
                for inNode in range(len(self.weights[layer])):
                    weightedSum = 0
                    nodeDeltas = []
                    for outNode in range(len(self.weights[layer][inNode])):
                        for k in range(len(self.nodeOutputs) - 1, layer + 1, -1):
                            for l in range(len(self.nodeOutputs[k])):
                                weightedSum += self.weights[k - 1][outNode][l] * (
                                    self.nodeOutputs[k][l] - expected[l]) * (
                                                   self.nodeOutputs[k][l] * (1 - self.nodeOutputs[k][l]))
                        nodeDeltas.append(weightedSum * (
                            (self.nodeOutputs[layer + 1][outNode] * (1 - self.nodeOutputs[layer + 1][outNode])) *
                            self.nodeOutputs[layer][inNode]))
                    layerDeltas.append(nodeDeltas)
                self.deltas.append(layerDeltas)

        for layer in range(len(self.weights)):  # weights
            for node in range(len(self.weights[layer])):
                for weight in range(len(self.weights[layer][node])):
                    self.weights[layer][node][weight] = self.weights[layer][node][weight] - (
                    self.deltas[-(layer + 1)][node][weight])

class Cards:
    suits = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
    values = range(2, 15)
    shoe = []

    def __init__(self, decks):
        self.deckSize = decks

    def newShoe(self):
        self.shoe = []
        for i in range(self.deckSize):
            for suit in self.suits:
                for number in self.values:
                    self.shoe.append((suit, number))
        shuffle(self.shoe)

    def getCardName(self, cards):  # takes in a list, do we need self here?
        output = []
        for card in cards:
            if card[1] <= 10:
                output.append((str(card[1]) + ' of ' + str(card[0]), card[1]))
            elif card[1] == 11:
                output.append(('Jack of' + ' ' + card[0], 10))
            elif card[1] == 12:
                output.append(('Queen of' + ' ' + card[0], 10))
            elif card[1] == 13:
                output.append(('King of' + ' ' + card[0], 10))
            elif card[1] == 14:
                output.append(('Ace of' + ' ' + card[0], 11))
        return output

    def deal(self, amount):  # creates a deck of cards
        dealCards = []
        if len(self.shoe) < 0.25 * (self.deckSize * 52):
            del self.shoe[:]
            self.newShoe()
        for i in range(amount):
            dealCards.append(self.shoe[0])
            del self.shoe[0]
        return dealCards

class Game:
    count = [0,0,0,0] #0 = win, 1= draw, 2 = lose, 3= blackjack count
    playerList = []
    agentCounter = 0
    money = 20000

    def __init__(self, numberOfPlayers, decks, hands, agent):
        self.agent = agent
        self.playerCount = numberOfPlayers
        self.cards = Cards(decks)
        self.handCount = hands
        self.setUpGame()

    def setUpGame(self): #set up a game of blackjack
        global split #splitting mode
        global double #doubling mode
        if self.agent == "thorpe":
            split = True
            double = True
        elif self.agent in ["random", "dealer", "hoyle"]:
            split = False
            double = False
        self.playerList = []
        dealer = Player("Dealer")
        self.playerList.append(dealer) #add dealer to the player list
        player = Player("Player")
        self.playerList.append(player) #add player to the player list
        self.hand()

    def split(self, player, pair): #split hand
        if len(player.playerCards)==1:
            hands = [[pair[0], pair[1]]]
        else:
            hands = [[pair[0], self.cards.deal(1)[0]], [pair[1], self.cards.deal(1)[0]]] #create 2 new hands with split cards

        for hand in hands:
            if len(player.playerCards) != 1:
                if pair[0][1] == 14: #Were the new hands created from split aces?
                    hand.append(2) #Yes, no more cards can be drawn
                else:
                    hand.append(1) #No, more cards can be drawn
            if hand[0][1] == hand[1][1]: #do the newly created hands contain a pair?
                playerCards = self.cards.getCardName(hand[:-1]) #find player cards
                pCard = playerCards[0][1]
                if self.agent == "nn": #decide whether or not to split new hand
                    decision = splitNN.propagate([pCard, self.playerList[0].score], 1)
                elif self.agent == "thorpe":
                    decision = self.thorpes([[pCard, 1, 0], self.playerList[0].score])
                    if decision == "P":
                        decision = 1
                    else:
                        decision = 0
                if decision >= 0.5:
                    if len(player.playerCards) == 1:
                        player.playerCards.append(0)
                    if printMode:
                        print("Player has decided to split")
                    self.split(player, hand)
                else:
                    if printMode:
                        print("Player has decided not to split")
                    if len(player.playerCards) != 1:
                        player.playerCards.append(hand)
            else:
                player.playerCards.append(hand)

    def dealerAction(self):
        dealer = self.playerList[0]
        dealerCards = self.cards.getCardName(dealer.playerCards[0][4:]) #get dealer cards
        if printMode:
            print("The dealer reveals his second card, it is the", dealerCards[1][0])  ##
            print("The dealer has a score of", dealer.playerCards[0][1], "\n")  ##
        player = self.playerList[1]
        for hand in player.playerCards: #draw cards while score <17
            if hand[1] < 22: #if player hand is not bust
                while (True):
                    if dealer.playerCards[0][1] < 17:
                        dealer.playerCards = [self.drawCard(dealer, dealer.playerCards[0])]
                    else:
                        break

    def thorpes(self, inputs):
        if inputs[0][1] == 1:
            inputs[0][0]= 2*inputs[0][0]
            action = thorpesPair[(10 * (inputs[1] - 2)) + int((inputs[0][0] / 2) - 2)]
        elif inputs[0][2] > 0:
            if inputs[0][0] > 20:
                inputs[0][0] = 20
            action = thorpesAce[(8 * (inputs[1] - 2)) + (inputs[0][0] - 13)]
        else:
            if inputs[0][0] > 17:
                inputs[0][0] = 17
            elif inputs[0][0] < 8:
                inputs[0][0] = 8
            action = thorpesSimple[(10 * (inputs[1] - 2)) + (inputs[0][0] - 8)]
        return action

    def hand(self):
        for i in range(self.handCount):
            for player in self.playerList:
                player.playerCards = []
                cards = self.cards.deal(2)  #deal cards to players
                if player.playerID != "Dealer" and splitTrain: #only deal pairs
                    cards = [cards[0], cards[0]]
                dealtCards = self.cards.getCardName(cards)
                player.playerCards.append(cards)#store player cards
                if player.playerID != "Dealer":
                    if printMode:
                        if dealtCards[0][1] == 11 and dealtCards [1][1] == 11: #if player has 2 aces in their hand
                            score = 12
                        else:
                            score =  dealtCards[0][1] + dealtCards[1][1]
                        print(player.playerID + "'s cards are the", dealtCards[0][0], "and the", dealtCards[1][0],
                                  ": Score =", score)
                else:
                    player.score= self.cards.getCardName(player.playerCards[0])[0][1]
                    if printMode:
                        print("\n\n\nThe " + player.playerID + "'s shown card is the", dealtCards[0][0], ": Score =",
                              player.score)

                if split and cards[0][1] == cards[1][1] and player.playerID != "Dealer": #split
                    self.split(player, cards)
                    if len(player.playerCards)>2: #if hand has been split
                        del player.playerCards[0:2] #delete split hand

                for hand in player.playerCards:
                    aces = 0
                    if hand[0][1] == 14:
                        aces += 1
                    if hand[1][1] == 14:
                        aces += 1
                    hand.insert(0, aces)  # insert into position 0 - ace count
                    hand.insert(1, 0)  # insert into position 1 - 0(current score count)
                    hand.insert(2, 0)  # insert into position 2 - 0: 0 = can't double down, 1 = Decided to double down, 2 = Has doubled down, 3 = decided not to double down
                    hand.insert(3, 0) # insert into position 3 - 0 : 0 = can't split, 1 = able to split, 2 = hand created from 2 aces splitting
                    if len(hand[4:])>2: #has the hand been split?
                        if hand[-1]==1:
                            hand[3]=1 #yes
                        else:
                            hand[3]=2 #yes and the hand originally held 2 aces
                        del hand[-1]
                    if hand[4][1] == hand[5][1]:  #Was splitting possible?
                        hand[3] = 1
                    dealtCards = self.cards.getCardName(hand[4:])

                    if hand[0] != 2: #does the hand not contain 2 aces?
                        hand[1] = dealtCards[0][1] + dealtCards[1][1]
                    else:
                        hand[1] = 12
                        hand[0] -= 1

                    if double == True and player.playerID != "Dealer" and hand[1]<21 and hand[3]!=2: #double down
                        if self.agent == "nn": #make decision on doubling down
                            decision = doubleNN.propagate([hand[1], self.playerList[0].score, hand[0]], 2)
                        elif self.agent == "thorpe":
                            decision = self.thorpes([[hand[1], 0, hand[0]], self.playerList[0].score])
                            if decision == "D" or decision == "Ds":
                                decision = 1
                            else:
                                decision = 0

                        if decision >= 0.5: #double down
                            if printMode:
                                print("player has decided to double down")
                            hand[2] = 1
                        else:
                            if printMode:
                                print("player has decided not to double down")
                            hand[2] = 3  # to signify that the option was there to double up

            for player in self.playerList[1:]:
                for hand in player.playerCards: #for each hand that the player holds
                    if self.agent == "nn" or self.agent == "thorpe":
                        while hand[1] < 21 and hand[2] != 2 and hand[3]!=2: #while cards can still be drawn
                            values = []
                            aceCount = hand[0]
                            for card in hand[4:]: #find current player score
                                if 9 < card[1] and card[1] < 14:
                                    values.append(10)
                                elif card[1] == 14:
                                    if aceCount > 0:
                                        values.append(11)
                                        aceCount -= 1
                                    else:
                                        values.append(1)
                                else:
                                    values.append(card[1])
                            playerScore = 0
                            for i in values:
                                playerScore += i
                            if self.agent == "nn":
                                decision = drawFoldNN.propagate([playerScore, self.playerList[0].score, hand[0]],0) #make decision on whether to draw another card
                                if hand[2] == 1: #############Review this
                                    decision = "D"
                            elif self.agent == "thorpe":
                                if hand[2]==1:
                                    decision = "D"
                                else:
                                    decision = self.thorpes([[playerScore, 0, hand[0]], self.playerList[0].score])
                                if decision == "H" or decision == "D":
                                    decision = 0
                                else:
                                    decision = 1
                            if decision <= 0.5:
                                hand = self.drawCard(player, hand)
                                if hand[2] == 1:
                                    hand[2] = 2
                            else:
                                if printMode:
                                    print("Player has decided to stand")
                                break

                    elif self.agent == "dealer": #if playing strategy = dealer's strategy
                        for player in self.playerList[1:]:
                            while hand[1] < 21:
                                if hand[1] > 16:
                                    break
                                else:
                                    hand = self.drawCard(player, hand)

                    elif self.agent == "hoyle": #if playing strategy = hoyle's strategy
                        for player in self.playerList[1:]:
                            while hand[1] < 21:
                                if hand[0] > 0:
                                    if hand[1] > 17:
                                        break
                                elif self.playerList[0].score < 7:
                                    if hand[1] > 12:
                                        break
                                else:
                                    if hand[1] > 16:
                                        break
                                hand = self.drawCard(player, hand)

                    elif self.agent == "random": #if playing strategy = random strategy
                        for player in self.playerList[1:]:
                            while hand[1] < 21:
                                rand = random.uniform(0, 1)
                                if rand >= 0.5:
                                    hand = self.drawCard(player, hand)
                                else:
                                    break

                #################################################################################
                self.dealerAction() #get dealer score
                for hand in player.playerCards: # find the result of each hand
                    best = (self.playerList[0].playerID,
                            self.playerList[0].playerCards[0][1]) #best hand is set to the dealer by default

                    #when player wins
                    if (self.playerList[0].playerCards[0][1] > 21 and hand[1] < 22) or (
                                    hand[1] > self.playerList[0].playerCards[0][1] and hand[1] < 22):
                        best = (player.playerID, hand[1])

                    #when player has 21
                    if hand[1] == 21 and self.playerList[0].playerCards[0][1] == 21: #do both players have 21?
                        if len(hand[4:]) == 2 and hand[3]<2: #does player have a blackjack?
                            if len(self.playerList[0].playerCards[0][4:]) == 2:#does dealer have a blackjack?(dealer=21)
                                best = ["draw", hand[1]] #the hands are tied
                            else:
                                best = ["blackjack", hand[1]] #player wins with a blackjack
                                self.count[3] += 1 #add 1 to the blackjack count
                        elif len(self.playerList[0].playerCards[0][4:])==2:#does dealer have a blackjack(player doesn't)
                            best = ["dealerWin", hand[1]] #dealer wins with a blackjack
                    elif hand[1] == 21 and len(hand[4:]) == 2 and hand[3]<2: #does player have a blackjack?(dealer!=21)
                        best = ["blackjack", hand[1]] #player wins with a blackjack
                        self.count[3] += 1 #add 1 to the blackjack count

                    if best[0] == self.playerList[0].playerID and best[1] == hand[
                        1] or best[0] == "draw": #if draw
                        if printMode:
                            print("The game was a draw")
                        self.count[1] += 1 #add 1 to the draw count
                        if self.agent == "nn":
                            if drawFoldTrain:
                                if hand[1] < 21:
                                    drawFoldNN.BP([0.5])
                            elif doubleTrain and hand[2] > 0:
                                doubleNN.BP([0.5])
                            elif splitTrain and hand[3] > 0:
                                splitNN.BP([0.5])
                    else:
                        if best[0] == self.playerList[1].playerID or best[0] == "blackjack": #if player wins
                            if printMode:
                                print(player.playerID, "wins with a score of", best[1])
                            if best[0]=="blackjack": #if blackjack then extra 0.5 returned
                                self.money += 0.5
                            self.count[0] += 1 #add 1 to the win count
                            self.money += 1
                            if hand[2] == 2:  # extra money and win as hand was doubled down
                                self.count[0] += 1
                                self.money += 1
                            if self.agent == "nn":
                                if drawFoldTrain:
                                    if best[1] < 21:
                                        drawFoldNN.BP([1])
                                elif splitTrain and hand[3] >0:
                                    splitNN.BP([1])
                                elif double and hand[2] > 0:
                                    if doubleTrain:
                                        doubleNN.BP([1])
                        else: #if dealer wins
                            if printMode:
                                print(self.playerList[0].playerID, "wins with a score of", best[1])
                            self.count[2] += 1 #add 1 to the lose count
                            self.money -= 1
                            if hand[2] == 2:  # Subtract extra from money and add to losses as hand was doubled down
                                self.count[2] += 1
                                self.money -= 1
                            if self.agent =="nn":
                                if drawFoldTrain:
                                    if hand[1] < 21:
                                        drawFoldNN.BP([0])
                                    elif hand[1] > 21:
                                        drawFoldNN.BP([1])
                                elif double and hand[2] > 0:
                                    if doubleTrain:
                                        doubleNN.BP([0])
                                elif splitTrain and hand[3]>0:
                                    splitNN.BP([0])

    def drawCard(self, player, hand): #deals a card to a player
        if printMode:
            print(player.playerID, "has decided to draw another card... \n")
        drawnCard = self.cards.deal(1)[0]
        if drawnCard[1] == 14: #if card is an ace
            hand[0] += 1 #add 1 to ace count
        hand.append(drawnCard) #append card to players hand
        dealtCard = self.cards.getCardName([drawnCard])
        previousTotal = hand[1]
        total = hand[1] + dealtCard[-1][1] #calculate new hand total
        if total > 21 and hand[0] > 0: #reduce value of any aces in the hand if score >21
            total -= 10
            hand[0] -= 1
        hand[1] = total
        if printMode:
            print("The", dealtCard[-1][0], "was dealt")
        if drawFoldTrain and player.playerID != "Dealer" and self.agent == "nn": #training drawFold neural network
            if total <= 21 and total > previousTotal:
                drawFoldNN.BP([0])
        return hand

def test():
    global  drawFoldTrain
    drawFoldTrain = False
    game = Game(1, 8, 50000, "nn")
    drawFoldTrain = True
    print((Game.count[0]+ (Game.count[3]*0.5))/(Game.count[0]+Game.count[1]+Game.count[2]))
    return game.count[0] / (game.count[0] + game.count[1] + game.count[2]) #(Game.count[0]+ (Game.count[3]*0.5))/(Game.count[0]+Game.count[1]+Game.count[2])

def currentWinMethod(winRate, weights):
    sorted = False
    global populationWinRate
    global populationWeights
    global currentWin
    for i in range(len(populationWinRate)-1, -1, -1):
        if winRate > populationWinRate[i]:
            saved = [populationWinRate[i], populationWeights[i]]
            populationWinRate[i] = winRate
            populationWeights[i] = weights
            if i ==0:
                sorted = True
            else:
                for k in range(i-1, -1, -1):
                    temp = [populationWinRate[k], populationWeights[k]]
                    populationWinRate[k]= saved[0]
                    populationWeights[k]= saved[1]
                    saved = temp
            if sorted == True:
                currentWin = populationWinRate[0]
                break

def GA(hands, population, generations, NN):
    global drawFoldNN
    global drawFoldTrain
    global split
    global splitTrain
    global double
    global doubleTrain
    global previousBestWin
    global previousBestWeights
    global overallWin
    global overallWeights
    global prevPopulationWeights
    global populationWeights
    global populationWinRate
    global currentWin
    global  populationStats

    if NN == drawFoldNN:
        drawFoldTrain = True
        split = False
        splitTrain = False
        double = False
        doubleTrain = False
    elif NN == splitNN:
        drawFoldTrain = False
        split = True
        splitTrain = True
        double = False
        doubleTrain = False
    elif NN == doubleNN:
        drawFoldTrain = False
        double = True
        doubleTrain = True
        split = False
        splitTrain = False

    overallWeights = []
    overallWin = 0

    results = test()
    print("original", results)

    for i in range(generations):  # change back to NN
        currentWin = 0
        #print("population best",populationWinRate[-1])
        #print("population stats", populationStats)
        populationStats = []
        prevPopulationWeights = populationWeights
        populationWinRate = [0,0,0]
        agentCounter = 0
        for n in range(population):  # population
            if i > 0 and n > 0:
                if agentCounter <10:
                    NN.weights = rando(prevPopulationWeights[0])
                elif agentCounter<20:
                    NN.weights = rando(prevPopulationWeights[1])
                else:
                    NN.weights = rando(prevPopulationWeights[2])
            Game(1, 8, hands, "nn")
            results = test()
            populationStats.append(results)
            print(results)
            if results >= currentWin:
                currentWinMethod(results, copy.deepcopy(NN.weights))
            agentCounter+=1
        previousBestWin = populationWinRate[-1]
        previousBestWeights = copy.deepcopy(populationWeights[-1])
        if previousBestWin >= overallWin:
            overallWin = previousBestWin
            overallWeights = copy.deepcopy(previousBestWeights)
    return overallWeights, overallWin

class Player:
    def __init__(self, ID):
        self.playerID = (ID)
        self.playerCards = []
        self.score = 0

def noBrac(arra):
    output = re.sub('[a-z()]', '', arra)
    return output

if __name__ == '__main__':
    start_time = time.time()
    hands = 50000
    population = 30
    generations = 15
    bestResultsArray = []

    drawFoldNN = NeuralNetwork((28, 2, 1), 0)
    splitNN = NeuralNetwork((10, 5, 1), 0)
    doubleNN = NeuralNetwork((28, 2, 1), 0)

    for nn in [drawFoldNN]:  # ,doubleNN, splitNN
        print("run")
        weights, winRate = GA(hands, population, generations, nn)
        bestResultsArray.append(winRate)
        print("best Weights", weights)
    print("best results", bestResultsArray)

    Game(1, 8, hands, "thorpe")
    print((Game.count[0])/(Game.count[0]+Game.count[1]+Game.count[2])) #+ (Game.count[3]*0.5)
    print((Game.count[1]) / (Game.count[0] + Game.count[1] + Game.count[2]))  # + (Game.count[3]*0.5)
    print((Game.count[2]) / (Game.count[0] + Game.count[1] + Game.count[2]))  # + (Game.count[3]*0.5)

    print(Game.count[3])