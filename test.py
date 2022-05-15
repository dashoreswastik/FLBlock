from web3 import Web3
url = "https://rinkeby.infura.io/v3/c50a17b9c8524402b3b355fa1cf0706c"
web3 = Web3(Web3.HTTPProvider(url))
print(web3.isConnected())

