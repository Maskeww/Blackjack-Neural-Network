import numpy as np
import time
import itertools
from random import shuffle
#enumerate isn't required, just loop for each item in the list

class NeuralNetwork: #Need to sort out where the class parameters should go

    def __init__(self): #rename hiddenNodeOutput etc #IS THE SELF. ACTUALLY REQUIRED?
        #network structure
        self.structure = (3,5,5,1) #Each number represents the weights in each layer of the NN
        self.weights = []
        self.nodeOutputs = [] #Outputs from each node in each layer
        self.nodeInputs = []
        self.deltas = []

        for i in range(len(self.structure)-1):
            self.weights.append(np.random.rand(self.structure[i], self.structure[i+1])) #Create and initialise weights

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sphere(self, inputTuple):
        answer=0
        if inputTuple[0] <0:
            input1 = -inputTuple[0]
        else:
            input1 = inputTuple[0]
        if inputTuple[1] <0:
            input2 = -inputTuple[1]
        else:
            input2 = inputTuple[1]
            answer += input1**2 + input2**2
        return answer

    def propagate(self, testSet):
        self.nodeInputs = []
        self.nodeOutputs = []
        for i in range(len(self.structure)):
            if i == 0:
                self.nodeInputs.append(testSet)
                self.nodeOutputs.append(self.nodeInputs[-1])
            elif i < len(self.structure)-1:
                self.nodeInputs.append(np.dot(self.nodeOutputs[-1], self.weights[i-1]))
                self.nodeOutputs.append(self.sigmoid(self.nodeInputs[-1]))
            else:
                self.nodeInputs.append(np.dot(self.nodeOutputs[-1], self.weights[i-1]))
                self.nodeOutputs.append(self.sigmoid(self.nodeInputs[-1]))
        return(self.nodeOutputs[-1])
        #print("=====ACTUAL OUTPUT=======", self.nodeOutputs[-1])

    def BP(self,expected):
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
                            self.nodeOutputs[-1][outNode] * (1 - self.nodeOutputs[-1][outNode])) * self.nodeOutputs[-2][
                                                  inNode]])  # WILL THIS GROUP THEM?
                self.deltas.append(lastLayer)
                layerDeltas = []

            elif current == len(self.weights) - 2:
                # for layer in range(current):
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
            else:
                node2Error = []
                intermediate = []

                for node0 in range(len(self.nodeOutputs[0])):
                    layerDeltas = []
                    for node1 in range(len(self.nodeOutputs[1])):
                        n1WeightedSum = 0
                        for node2 in range(len(self.nodeOutputs[2])):
                            weightedSum = 0
                            for node3 in range(len(self.nodeOutputs[3])):
                                weightedSum += (self.nodeOutputs[-1][node3] - expected[node3]) * (
                                self.nodeOutputs[-1][node3] * (1 - (self.nodeOutputs[-1][node3])) *
                                self.weights[-1][node2][node3])
                            node2Error.append(weightedSum)
                        for weights in range(len(self.weights[-2][node1])):
                            n1WeightedSum += self.weights[-2][node1][weights] * node2Error[weights] * (
                            self.nodeOutputs[2][weights] * (1 - self.nodeOutputs[2][weights]))
                        layerDeltas.append(
                            n1WeightedSum * (self.nodeOutputs[1][node1]) * (1 - self.nodeOutputs[1][node1]) *
                            self.nodeOutputs[0][node0])
                    intermediate.append(layerDeltas)
                self.deltas.append(intermediate)

        for layer in range(len(self.weights)):  # weights
            for node in range(len(self.weights[layer])):
                for weight in range(len(self.weights[layer][node])):
                    self.weights[layer][node][weight] = self.weights[layer][node][weight] - (self.deltas[-(layer + 1)][node][weight])

class Cards:
    suits = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
    values = range(2, 15)
    shoe = []
    discardPile = []


    def __init__(self, decks):
        self.deckSize = decks
        self.newShoe()


    def newShoe(self):
        for i in range(self.deckSize):
            for Card in itertools.product(self.suits,self.values):
                self.shoe.append(Card) #Cartesian Product to Create Deck
        shuffle(self.shoe)

    def getCardName(self, cards): # takes in a list, do we need self here?
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

    def deal(self, amount): #returns a list of cards
        dealCards = []
        for i in range(amount):
            dealCards.append(self.shoe[0])
            del self.shoe[0]
        return dealCards

class Game:
    winCount = 0
    drawCount = 0
    loseCount = 0
    playerList = []

    def __init__(self, numberOfPlayers,decks, hands):
        self.playerCount = numberOfPlayers
        self.cards = Cards(decks)
        self.handCount = hands
        self.setUp()

    def setUp(self):
        player = Player("Dealer")
        self.playerList.append(player)
        for i in range(self.playerCount):
            player = Player("Player "+ str(i))
            self.playerList.append(player)
        self.hand()

    def hand(self):
        for i in range(self.handCount):
            if len(self.cards.shoe)< 0.25*(self.cards.deckSize * 52):
                del self.cards.shoe[:]
                self.cards.newShoe()
            for player in self.playerList:
                cards =  self.cards.deal(2) #deal cards to players -> [("diamonds", 10), ("diamonds", 9)]
                if cards[0][1]==14:
                    player.aces+=1
                if cards[1][1]==14:
                    player.aces+=1
                player.playerCards = cards
                dealtCards = self.cards.getCardName(player.playerCards) #could save this as a seperate variable in player class
                if player.aces!=2:
                    player.playerTotal = dealtCards[0][1] + dealtCards[1][1]
                else:
                    player.playerTotal = 12
                    player.aces-=1
                if player.playerID !="Dealer":
                    #print(player.playerID + "'s cards are the", dealtCards[0][0],"and the", dealtCards[1][0], ": Score =", player.playerTotal)##
                    pass
                else:
                    #print("The " + player.playerID + "'s shown card is the", dealtCards[0][0], ": Score =", self.cards.getCardName(player.playerCards)[0][1])##
                    pass

            for player in self.playerList[1:]:
                while player.playerTotal <21:
                    decision = nn.propagate([player.playerTotal, self.playerList[0].playerTotal, player.aces])  # feeds the input through the network
                    #decision = input(player.playerID + " your score is " + str(player.playerTotal) + ". Would you like to twist? Y = Yes, N= No ")
                    if decision >= 0.5:
                        self.twist(player)
                    else:
                        break
            if player.playerTotal>21:
                #print(player.playerID + "'s total is", player.playerTotal, "BUST!")##
                nn.BP([0])
            else:
                #print((player.playerID + "'s total is " + str(player.playerTotal) + "\n"))##
                nn.BP([1])

            dealer = self.playerList[0]
            dealerCards = self.cards.getCardName(dealer.playerCards)#cards object required?
            #print("The dealer reveals his second card, it is the", dealerCards[1][0])##
            #print("The dealer has a score of", dealer.playerTotal, "\n")##

            for player in self.playerList[1:]:
                if player.playerTotal <22:
                    while(True):
                        if dealer.playerTotal<17 and dealer.playerTotal<=player.playerTotal:
                            self.twist(dealer)
                        else:
                            break

            best = ("The Dealer", 0) #Code under here could be more efficient
            for player in self.playerList:
                if player.playerTotal > best[1] and player.playerTotal<22:
                    best = (player.playerID, player.playerTotal)

            if best[0] == self.playerList[0].playerID and best[1] == player.playerTotal:
                #print("The game was a draw")##
                self.drawCount+=1
                if self.playerList[1].playerTotal <21:
                    nn.BP([0.5])
            else:
                #print(best[0], "wins with a score of", best[1])##
                if best[0] == self.playerList[1].playerID:
                    self.winCount+=1
                    nn.BP([0])
                else:
                    self.loseCount+=1
                    if self.playerList[1].playerTotal <22:
                        nn.BP([1])

    def twist(self, player):
        #print(player.playerID, "has decided to twist... \n")##
        card = self.cards.deal(1)[0]
        if card[1] == 14:
            player.aces+=1

        player.playerCards.append(card)
        dealtCards = self.cards.getCardName(player.playerCards)
        total = player.playerTotal + dealtCards[-1][1]
        if total > 21 and player.aces > 0:
            total-=10
            player.aces-=1
        player.playerTotal = total
        #print("The", dealtCards[-1][0], "was dealt")##

class Player:
    def __init__(self, ID):
        self.playerID = (ID)
        self.playerCards = []
        self.playerTotal = 0
        self.aces = 0

if __name__ == '__main__':
    start_time = time.time()
    nn = NeuralNetwork()
    game = Game(1,6,1000000)

    print("Win Count:", game.winCount)
    print("Lose Count:", game.loseCount)
    print("Draw Count:", game.drawCount)
    print("win percentage:",str(game.winCount/(game.winCount+game.loseCount+game.drawCount)*100)+ "%")

    print("--- %s seconds ---" % (time.time() - start_time))
