import time
import hashlib
import pickle

class Block:
    def __init__(self,client_ID, weights, timestamp = time.time(), previous_hash = None):
        self.clientID = client_ID
        self.weights = weights
        self.timestamp = timestamp
        self.previous_hash = previous_hash


class BlockChain:
    def __init__(self) -> None:
        self.map = {}
        self.genesis = True
        self.previous_hash = None

    def make_block(self,clientID, weights):
        
        if self.genesis == True:
            b = Block(clientID, weights)
            self.genesis = False
            # w = '\n'.join('\t'.join('%0.3f' %x for x in y) for y in weights)
            text = str(clientID)+str(b.timestamp)
            self.previous_hash = self.make_hash(text)
            self.map[self.previous_hash] = b

        else:
            b = Block(clientID, weights, previous_hash=self.previous_hash)
            # w = '\n'.join('\t'.join('%0.3f' %x for x in y) for y in weights)
            text = str(clientID)+str(b.timestamp)+str(self.previous_hash)
            self.previous_hash = self.make_hash(text)
            self.map[self.previous_hash] = b

        return self.previous_hash
           

    def make_hash(self,text):
        return hashlib.sha256(text.encode()).hexdigest()



# blockchain = BlockChain()

# clientID = "abcd"
# weights = [[1,2,3],[4,5,6]]
# samples = 1000


# prev_hash = blockchain.make_block(clientID,weights,samples)
# prev_hash = blockchain.make_block("xyz",[[0,0,0],[0,0,0]],1000)

# print(blockchain.map)

# with open('blockchain.pickle', 'wb') as handle:
#     pickle.dump(blockchain, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('blockchain.pickle', 'rb') as handle:
#     b = pickle.load(handle)

# print(b.map)



