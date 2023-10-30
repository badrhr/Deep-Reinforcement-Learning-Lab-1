from agent.agent import Agent
from functions import *
import sys

total_profitl = []
buy_info = []
sell_info = []
data_Store = []

stock_name, window_size, episode_count = 'GOLD', 3, 10

agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):

	print("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size + 1)

	total_profit = 0
	agent.inventory = []

	for t in range(l):
        #Sample a Random action in the first episodes 
        #and then try to predict the best action for a given state
		action = agent.act(state)

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(data[t])
			print("Buy: " + formatPrice(data[t]))
            
            #save results for visualisation
			buy_info.append(data[t])
			d = str(data[t]) + ', ' + 'Buy'
			data_Store.append(d)
            
		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price
            
			print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
			total_profitl.append(data[t] - bought_price)
            
			step_price = data[t] - bought_price
            
			info = str(data[t]) +',' + str( step_price) +',' + str(reward)
			sell_info.append(info)
			d = str(data[t]) + ', ' + 'Sell'
			data_Store.append(d)



		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print("--------------------------------")
			print("Total Profit: " + formatPrice(total_profit))
			print("--------------------------------")

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 10 == 0:
		agent.model.save("models/model_ep" + str(e))