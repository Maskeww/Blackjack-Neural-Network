import time
import itertools
from random import shuffle

#enumerate isn't required, just loop for each item in the list

class Cards:
    suits = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
    values = range(2, 15)
    shoe = []
    discardPile = []
    def __init__(self, decks):
        for i in range(decks):
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
                    print(player.playerID + "'s cards are the", dealtCards[0][0],"and the", dealtCards[1][0], ": Score =", player.playerTotal)
                else:
                    print("The " + player.playerID + "'s shown card is the", dealtCards[0][0], ": Score =", self.cards.getCardName(player.playerCards)[0][1])

            for player in self.playerList[1:]:
                while player.playerTotal <21:
                    decision = input(player.playerID + " your score is " + str(player.playerTotal) + ". Would you like to twist? Y = Yes, N= No ")
                    if decision == "Y":
                        self.twist(player)
                    elif decision == "N":
                        break
                    else:
                        print("Please enter a valid character (Y or N) ")
            if player.playerTotal>21:
                print(player.playerID + "'s total is", player.playerTotal, "BUST!")
            else:
                print((player.playerID + "'s total is " + str(player.playerTotal) + "\n"))

            dealer = self.playerList[0]
            dealerCards = self.cards.getCardName(dealer.playerCards)#cards object required?
            print("The dealer reveals his second card, it is the", dealerCards[1][0])
            print("The dealer has a score of", dealer.playerTotal, "\n")

            for player in self.playerList[1:]:
                if player.playerTotal <22:
                    while(True):
                        if dealer.playerTotal<18 and dealer.playerTotal<=player.playerTotal:
                            self.twist(dealer)
                        else:
                            break

            best = ("The Dealer", 0) #Code under here could be more efficient
            for player in self.playerList:
                if player.playerTotal > best[1] and player.playerTotal<22:
                    best = (player.playerID, player.playerTotal)

            if best[0] == self.playerList[0].playerID and best[1] == player.playerTotal:
                print("The game was a draw")
                self.drawCount+=1
            else:
                print(best[0], "wins with a score of", best[1])
                if best[0] == self.playerList[1].playerID:

                    self.winCount+=1
                else:
                    self.loseCount+=1
            print(self.winCount, "Win Count")
            print(self.loseCount, "Lose Count")
            print(self.drawCount, "Draw Count")

    def twist(self, player):
        print(player.playerID, "has decided to twist... \n")
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
        print("The", dealtCards[-1][0], "was dealt")

class Player:
    def __init__(self, ID):
        self.playerID = (ID)
        self.playerCards = []
        self.playerTotal = 0
        self.aces = 0

if __name__ == '__main__':
    start_time = time.time()
    game = Game(1,6,5)
    print("--- %s seconds ---" % (time.time() - start_time))