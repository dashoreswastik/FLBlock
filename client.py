from web3 import Web3
import json
# from web3.middleware import geth_poa_middleware


url = "https://rinkeby.infura.io/v3/c50a17b9c8524402b3b355fa1cf0706c"

web3 = Web3(Web3.HTTPProvider(url))
# web3.middleware_onion.inject(geth_poa_middleware, layer=0)

print("Connected to Rinkeyby:",web3.isConnected())

with open('abi.json','r') as f:
    abi = json.load(f)

address = "0x383D424e1639D89b5883D4D782b67ef0C98DF5F2"

contract = web3.eth.contract(address=address,abi=abi)

txhash = contract.functions.set_weights("1234",1,"xyz").transact()

web3.eth.waitForTransactionReciept(txhash)

print("Sent Weights")


