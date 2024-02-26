import gym
import gymnasium
from gym import spaces
import pandas as pd
import numpy as np
from datetime import timedelta

# This environment simulates a cryptocurrency trading scenario where an agent can take positions in Bitcoin (BTC) and Ethereum (ETH)
# The state space includes positions (Long, Short, Cash, Cash at Risk, ETH and BTC holdings), facilitating a complex trading strategy simulation.
# The environment is designed to challenge the agent to learn not just trading strategies but also risk management, considering transaction costs and market liquidity.

#Include or not include regime really tricky, the HMM is best here because it would be 100% accurate but the agent would
#Not learn the error of the radar systems


#State space [Long, Short, Cash, cash_AT_RISK, ETH_AT_HAND,BTC_AT_HAND]

#long: Close, Volume of 14 days
#Short: All columns of 6 hours
#Cash: Realistically maybe we count the totally converted cash so if cash < 1 we lost money here
#cash_AT_RISK: Percentage of Cash in BTC or ETH essientally a liquiditiy counter
#Amount of ETH out of the total percentage
#Amount of BTC out of the total percentage

class CryptoTradingEnv(gymnasium.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, df_btc, df_eth, df_btc_eth, INIT_cash, episode_length=20160, start_point=None, end_point=None, transaction_cost=0.0, Pass_out = 0,noise = False):
        super(CryptoTradingEnv, self).__init__()

        # Check for matching datetime indices and data integrity
        self._validate_data(df_btc, df_eth, df_btc_eth)

        # Load data
        self.df_btc = df_btc
        self.df_eth = df_eth
        self.df_btc_eth = df_btc_eth

        self.cash = INIT_cash
        self.cash_AT_RISK = 0 #percentage of cash in ETH or BTC
        self.cash_AT_HAND = INIT_cash
        self.ETH_AT_HAND = 0 #Quantity of BTC
        self.BTC_AT_HAND = 0 #Quantity of ETH

        self.PCT_CASH = 1
        self.PCT_BTC = 0
        self.PCT_ETH = 0

        self.transaction_cost = transaction_cost
        self.episode_length = episode_length
        self.start_point = start_point
        self.end_point = end_point
        self.reward = 0
        self.operations_counter = 0
        # Initialize or reset any state variables or portfolios
        self.reset_state_variables()
        self.Pass_out = Pass_out #chances of agent losing control this turn, 0 being none 0.1 being 10%

        self.noise = noise #adding OU noise equal to the variance
        # Define action and observation space
        # BUY/SELL BTC, BUY/SELL ETH, CONVERT BTC/ETH,
        self.action_space = spaces.Box(low=np.array([-1,-1,-1]), high=np.array([1,1,1]), dtype=np.float16)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def _validate_data(self, df_btc, df_eth, df_btc_eth):
        # Check for matching datetime indices across datasets
        # Ensure datetime indices match across BTC, ETH, and conversion rate data frames to synchronize market data.
        if not (df_btc.index.equals(df_eth.index) and df_btc.index.equals(df_btc_eth.index)):
            raise ValueError("Datetime indices of BTC, ETH, and BTC-ETH conversion tables do not match.")

        # Check for NaN, Inf values
        if df_btc.isnull().values.any() or df_eth.isnull().values.any() or df_btc_eth.isnull().values.any():
            raise ValueError("One or more tables contain NaN values.")

        if np.isinf(df_btc.values).any() or np.isinf(df_eth.values).any() or np.isinf(df_btc_eth.values).any():
            raise ValueError("One or more tables contain Inf values.")

        # Optional: Check for missing timestamps if the datasets are expected to be continuous without gaps

    def reset_state_variables(self):
        self.cash_AT_RISK = 0  # percentage of cash in ETH or BTC
        self.cash_AT_HAND = INIT_cash
        self.ETH_AT_HAND = 0  # Quantity of BTC
        self.BTC_AT_HAND = 0  # Quantity of ETH

        self.PCT_CASH = 1
        self.PCT_BTC = 0
        self.PCT_ETH = 0
        self.reward = 0
        self.operations_counter = 0
        # Initialize or reset other necessary state variables here

    def step(self, action):
        # Extract individual actions for clarity
        action_btc, action_eth, action_convert = action

        # Execute trade actions
        self.BUY('BTC', action_btc)
        self.BUY('ETH', action_eth)
        self.Convert(action_convert)

        # Update the market step
        self.current_step += 1
        if self.current_step >= self.end_point:
            done = True
        else:
            done = False

        # Calculate reward
        reward = self.calculate_reward()

        # Get next observation
        observation = self._get_observation()

        # Additional info can include any auxiliary data useful for debugging or analysis
        info = {}

        return observation, reward, done, info


    #take a closer look at the volume information columns
    #cash buy
    def BUY(self, target, amount):
        # Check if the amount is within [0, 1] range
        if amount < 0 or amount > 1:
            self.reward -= 5
            return  # Ignore invalid amounts

        # Calculate the cash to be used for the purchase, avaliable cash * % allocated
        cash_to_use = self.cash_AT_HAND * amount

        # Get current price from the data
        if target == 'BTC':
            current_price = self.df_btc.iloc[self.current_step]['Close']
            crypto_bought = cash_to_use / current_price
            if crypto_bought < 0.001:
                self.reward -= 5
                return  # Do not proceed with the transaction
            self.BTC_AT_HAND += crypto_bought
            self.cash_AT_HAND -= self.transaction_cost
            self.operations_counter += 1
        elif target == 'ETH':
            current_price = self.df_eth.iloc[self.current_step]['Close']
            crypto_bought = cash_to_use / current_price
            if crypto_bought < 0.001:
                self.reward -= 5
                return  # Do not proceed with the transaction
            self.BTC_AT_HAND += crypto_bought
            self.cash_AT_HAND -= self.transaction_cost
            self.operations_counter += 1

        #subtract the cash
        self.cash_AT_HAND -= cash_to_use

    #cash sell
    #delayed one step to encourage anticpation
    def SELL(self, target, amount):
        # Fetch current market data

        # If SELL amount * Price < trading fee, massive punishment
        # Calculate the value of the assets the agent wants to sell
        if target == 'BTC':
            current_volume = self.df_btc.iloc[self.current_step]['volume']
            current_price = self.df_btc.iloc[self.current_step]['Close']
            asset_to_sell = self.BTC_AT_HAND * amount
            asset_value = asset_to_sell * current_price
        elif target == 'ETH':
            current_volume = self.df_eth.iloc[self.current_step]['volume']
            current_price = self.df_eth.iloc[self.current_step]['Close']
            asset_to_sell = self.ETH_AT_HAND * amount
            asset_value = asset_to_sell * current_price

        market_liquidity = current_volume * current_price  # Estimate of market liquidity
        # Estimate slippage based on the order's size relative to market liquidity
        slippage_factor = self.estimate_slippage(asset_value / market_liquidity)

        # Adjust the sell price to account for slippage
        effective_sell_price = current_price * (1 - slippage_factor)

        # Update holdings and cash
        if target == 'BTC':
            self.BTC_AT_HAND -= asset_to_sell
        elif target == 'ETH':
            self.ETH_AT_HAND -= asset_to_sell

        self.cash_AT_HAND += asset_value * (1 - slippage_factor)

    def estimate_slippage(self, order_to_liquidity_ratio):
        # Simple slippage model: linear increase in slippage with order size
        base_slippage = 0.005  # Base slippage for small orders
        slippage = base_slippage + order_to_liquidity_ratio
        return min(slippage, 0.05)  # Cap the slippage to a maximum of 5%

    def Convert(self, direction: float) -> float:
        """Converts between ETH and BTC.

        Args:
            direction: A float representing the conversion amount. If negative,
                       converts a percentage of ETH into BTC. If positive,
                       converts BTC into ETH.

        Returns:
            none this is a setter


        Raises:
             TypeError: If 'direction' is not a float.
        """

        if not isinstance(direction, float):
            raise TypeError("Direction must be a float")
        if direction > 0:
            #logic to convert ETH to BTC
        if direction < 0:

        if direction == 0:
            return
        # ... (Implement your conversion logic here)
    # First lets sort all the DF
    # If no start point and no end point,perform the necessary check and assign randomly start point and end point
    # If there is a start point and no end point, we ensure there is enough space size epsoide_length and assign end point
    def update_relative_position(self):
        # Calculate current portfolio value
        current_value = (
                self.BTC_AT_HAND * self.df_btc.iloc[self.current_step]['Close'] +
                self.ETH_AT_HAND * self.df_eth.iloc[self.current_step]['Close'] +
                self.cash
        )
        initial_value = self.INIT_cash  # Assuming INIT_cash is the initial total value
        # Calculate net change in value
        net_change = current_value - initial_value
        # Update relative position (as a percentage or absolute value)
        self.relative_position = net_change / initial_value
    def reset(self):
        # Prepare the environment for a new episode, including data sorting and validation of the episode's feasibility given the data.
        # Sort the data frames to ensure chronological order.
        # Sort all the DataFrames by their index (assuming datetime index)
        self.df_btc.sort_index(inplace=True)
        self.df_eth.sort_index(inplace=True)
        self.df_btc_eth.sort_index(inplace=True)

        # Ensure there's enough data for the lookback period and the episode length
        total_required_data_points = 20160 + self.episode_length  # 2 weeks lookback + episode length

        # Check if the dataset has enough data to meet the requirement
        if len(self.df_btc) < total_required_data_points:
            raise ValueError("Dataset does not have enough data for the required lookback period and episode length.")

        # Handle cases based on provided start and end points
        if self.start_point is None and self.end_point is None:
            # Adjust max_start_point to leave space for the lookback period and future data
            max_start_point = len(self.df_btc) - total_required_data_points
            # Randomly assign start point, ensuring space for lookback
            self.start_point = np.random.randint(0, max_start_point) + 20160
            # Calculate and assign end point based on episode length
            self.end_point = self.start_point + self.episode_length - 20160
        elif self.start_point is not None and self.end_point is None:
            # Ensure the start point allows for a full episode and lookback
            if self.start_point < 20160 or self.start_point + self.episode_length > len(self.df_btc):
                raise ValueError("Start point does not allow for a full episode and required lookback period.")
            self.end_point = self.start_point + self.episode_length - 20160
        # If both start and end points are provided, or just the end point is provided, additional validation can be added here

        self.current_step = self.start_point

        # Initialize or reset any state variables or portfolios
        self.reset_state_variables()

        return self._get_observation()


    def _get_observation(self):
        # Return the state based on the current step
        # Implement logic to return Long or Short state as specified
        return np.array([0])  # Placeholder

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError("Only console mode is supported at the moment")

        # Example of information to print. Customize it based on your needs.
        print(f"Step: {self.current_step}")
        print(f"Current Timeframe Start: {self.start_point + self.current_step}")
        print("Current Portfolio State: [Placeholder for portfolio state]")
        print("Recent Action: [Placeholder for last action taken]")
        print("Current Observation: [Placeholder for current observation]")
        # Add any other relevant information you wish to track

    def expert(self):
        #This run the same time stamp and state space but using the both, return
        #Best_bot
        #rewards
        #return Best_bot, rewards
        pass

    def cheat(self, forward_length):
        #Human hindsight
        pass



# Testing script
if __name__ == "__main__":
    # Mock data for testing without loading actual CSV files
    mock_data_length = 10000000  # Equivalent to 1 week of minute-ticks
    df_btc = pd.DataFrame({
        'Close': np.random.rand(mock_data_length),
        'Volume': np.random.rand(mock_data_length),
    })
    df_eth = df_btc.copy()  # Simplification for testing
    df_btc_eth = df_btc.copy()  # Simplification for testing

    env = CryptoTradingEnv(df_btc, df_eth, df_btc_eth,10000)
    observations = env.reset()

    for _ in range(3):  # Simulate 3 steps
        action = env.action_space.sample()  # Sample a random action
        observations, rewards, done, info = env.step(action)
        env.render()  # Print the current state to console

        if done:
            break

# Example CSV loading
# df_btc = pd.read_csv('path_to_your_btc_usd_data.csv')
# df_eth = pd.read_csv('path_to_your_eth_usd_data.csv')
# df_btc_eth = pd.read_csv('path_to_your_btc_eth_data.csv')

# Initialize the environment
# env = CryptoTradingEnv(df_btc, df_eth, df_btc_eth)
