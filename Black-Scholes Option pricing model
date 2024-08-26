## Black-Scholes Option pricing model
    from pydantic import BaseModel, Field, computed_field
    from typing import Literal
    import numpy as np
    from scipy.stats import norm
    from datetime import datetime

    import matplotlib.pyplot as plt
    from tabulate import tabulate

    import opstrat as op
    import yfinance as yf
    import pandas as pd


    class OptionInputs(BaseModel):
        option_type: Literal["call", "put"]
        spot_price: float = Field(gt=0, description="Current price of the underlying asset")
        strike_price: float = Field(gt=0, description="Strike price of the option")
        time_to_expiry: float = Field(gt=0, description="Time to expiration in years")
        risk_free_rate: float = Field(ge=0, le=1, description="Risk-free interest rate")
        volatility: float = Field(gt=0, description="Volatility of the underlying asset")
        
    class BlackScholesModel(BaseModel):
        inputs: OptionInputs
        
        @computed_field
        def d1(self) -> float:
            return (np.log(self.inputs.spot_price / self.inputs.strike_price) + 
                    (self.inputs.risk_free_rate + 0.5 * self.inputs.volatility**2) * self.inputs.time_to_expiry) / \
                   (self.inputs.volatility * np.sqrt(self.inputs.time_to_expiry))
        
        @computed_field
        def d2(self) -> float:
            return(self.d1 - self.inputs.volatility * np.sqrt(self.inputs.time_to_expiry))
    
        @computed_field
        def price(self) -> float:
            if self.inputs.option_type == "call":
                return self.inputs.spot_price * norm.cdf(self.d1) - self.inputs.strike_price * np.exp((-self.inputs.risk_free_rate)*self.inputs.time_to_expiry)*norm.cdf(self.d2)
            else:
                return self.inputs.strike_price * np.exp((-self.inputs.risk_free_rate)*self.inputs.time_to_expiry)*norm.cdf(-self.d2)-self.inputs.spot_price * norm.cdf(-self.d1)

        
        @computed_field
        def delta(self) -> float:
            if self.inputs.option_type == "call":
                return norm.cdf(self.d1)
            else:
                return norm.cdf(self.d1) - 1
            
        @computed_field
        def gamma(self) -> float:
            return norm.pdf(self.d1)/(self.inputs.spot_price*self.inputs.volatility*np.sqrt(self.inputs.time_to_expiry))
        
        @computed_field
        def vega(self) -> float:
            return self.inputs.spot_price * norm.pdf(self.d1) * np.sqrt(self.inputs.time_to_expiry) / 100
        
        @computed_field
        def theta(self) -> float:
            common = -(self.inputs.spot_price * norm.pdf(self.d1) * self.inputs.volatility) / (2 * np.sqrt(self.inputs.time_to_expiry))
            if self.inputs.option_type == "call":
                return (common - self.inputs.risk_free_rate * self.inputs.strike_price * 
                        np.exp(-self.inputs.risk_free_rate * self.inputs.time_to_expiry) * norm.cdf(self.d2)) / 365
            else:  # put option
                return (common + self.inputs.risk_free_rate * self.inputs.strike_price * 
                        np.exp(-self.inputs.risk_free_rate * self.inputs.time_to_expiry) * norm.cdf(-self.d2)) / 365
        
        @computed_field
        def rho(self) -> float:
            if self.inputs.option_type == "call":
                return self.inputs.strike_price * self.inputs.time_to_expiry * np.exp(-self.inputs.risk_free_rate * self.inputs.time_to_expiry) * norm.cdf(self.d2) / 100
            else:  # put option
                return -self.inputs.strike_price * self.inputs.time_to_expiry * np.exp(-self.inputs.risk_free_rate * self.inputs.time_to_expiry) * norm.cdf(-self.d2) / 100
    
    
    
    call_option_inputs = OptionInputs(
        option_type="call",
        spot_price=100,
        strike_price=100,
        time_to_expiry=1,
        risk_free_rate=0.05,
        volatility=0.2
    )
    
    # Create a copy of the inputs with the option_type set to "put"
    put_option_inputs = call_option_inputs.model_copy(update={"option_type": "put"})
    
    # Creating a BlackScholesModel object with the inputs
    call_option = BlackScholesModel(inputs=call_option_inputs)
    put_option = BlackScholesModel(inputs=put_option_inputs)
    
    
    header = ['Option Type', 'Option Price', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
    table = [
        [call_option_inputs.option_type, call_option.price, call_option.delta, call_option.gamma, call_option.theta, call_option.vega, call_option.rho],
        [put_option_inputs.option_type, put_option.price, put_option.delta, put_option.gamma, put_option.theta, put_option.vega, put_option.rho]
    ]
    
    print(tabulate(table,header))    
    
    
    
    
    #Single plotter (Long call)
    op.single_plotter(spot=call_option_inputs.spot_price, strike=call_option_inputs.strike_price, op_type = 'c', tr_type='b', op_pr = call_option.price, spot_range=20)
    
    
    #MShort Straddle
    leg1 = BlackScholesModel(inputs=OptionInputs(option_type="call", spot_price=100, strike_price=100, time_to_expiry=1,risk_free_rate=0.05,volatility=0.2))
    leg2 = BlackScholesModel(inputs=OptionInputs(option_type="put", spot_price=100, strike_price=100, time_to_expiry=1,risk_free_rate=0.05,volatility=0.2))
    
    # The particulars of each option has to be provided as a list of dictionaries.
    op_1 = {'op_type': 'c', 'strike':leg1.inputs.strike_price, 'tr_type': 's', 'op_pr': leg1.price}
    op_2 = {'op_type': 'p', 'strike':leg2.inputs.strike_price, 'tr_type': 's', 'op_pr': leg2.price}
    op_list = [op_1, op_2]
    
    # Multi-plotter
    op.multi_plotter(spot=leg1.inputs.spot_price, spot_range=50, op_list=op_list)
    
    
    
    #Short Strangle
    leg1 = BlackScholesModel(inputs=OptionInputs(option_type="call",spot_price=100,strike_price=110,time_to_expiry=1,risk_free_rate=0.05,volatility=0.2))
    leg2 = BlackScholesModel(inputs=OptionInputs(option_type="put",spot_price=100,strike_price=95,time_to_expiry=1,risk_free_rate=0.05,volatility=0.2))
    
    # The particulars of each option has to be provided as a list of dictionaries.
    op_1 = {'op_type': 'c', 'strike':leg1.inputs.strike_price, 'tr_type': 's', 'op_pr': leg1.price}
    op_2 = {'op_type': 'p', 'strike':leg2.inputs.strike_price, 'tr_type': 's', 'op_pr': leg2.price}
    op_list = [op_1, op_2]
    
    # Multi-plotter
    op.multi_plotter(spot=leg1.inputs.spot_price, spot_range=30, op_list=op_list)
    
    
    
    
  ### BSM model for SPY oprion prices
    
    spy = yf.Ticker('SPY')
    options = spy.option_chain('2024-09-30')
    
    dte = (datetime(2024, 9, 30) - datetime.today()).days/365
    spot = 532.905; strike = 533; rate = 0.00; dte = dte; vol = 0.2107
    
    spy_opt =BlackScholesModel(inputs=OptionInputs(option_type="call",spot_price=spot,strike_price=strike,time_to_expiry=dte,risk_free_rate=rate,volatility=vol))
    print(f'Option Price of SPY240930C00533000 with BS Model is {spy_opt.price:0.4f}')
    
    df = options.puts[(options.puts['strike']>=450) & (options.puts['strike']<=600)]
    df.reset_index(drop=True, inplace=True)
    
    # Dataframe manipulation with selected fields
    df = pd.DataFrame({'Strike': df['strike'], 
                       'Price': df['lastPrice'], 
                       'ImpVol': df['impliedVolatility']})
    
    # Derive greeks and assign to dataframe as columns
    df['Delta'] = df['Gamma'] = df['Vega'] = df['Theta'] = 0.
    
    for i in range(len(df)):
    
        option = BlackScholesModel(inputs=OptionInputs(option_type="call",spot_price=spot,strike_price=df['Strike'].iloc[i],time_to_expiry=dte,risk_free_rate=rate,volatility=df['ImpVol'].iloc[i]))
        
        df['Delta'].iloc[i] = option.delta
        df['Gamma'].iloc[i] = option.gamma
        df['Vega'].iloc[i] = option.vega
        df['Theta'].iloc[i] = option.theta
    
    # Check output
    df.head(2)
    
  ## Data Visualisation
    fig, ax = plt.subplots(2,2, figsize=(20,10))
    
    ax[0,0].plot(df['Strike'], df['Delta'], color='r', label='SEP 24')
    ax[0,1].plot(df['Strike'], df['Gamma'], color='b', label='SEP 24')
    ax[1,0].plot(df['Strike'], df['Vega'],  color='k', label='SEP 24')
    ax[1,1].plot(df['Strike'], df['Theta'], color='g', label='SEP 24')
        
    # Set axis title
    ax[0,0].set_title('Delta'), ax[0,1].set_title('Gamma'), ax[1,0].set_title('Vega'), ax[1,1].set_title('Theta')
    
    # Define legend
    ax[0,0].legend(), ax[0,1].legend(), ax[1,0].legend(), ax[1,1].legend()
    
    # Set title
    fig.suptitle('Greeks Vs Strike')
    
    plt.show()
    




