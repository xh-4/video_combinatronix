"""
Trading System Example using ChannelPy

This example demonstrates how to use channel algebra for financial trading
with adaptive thresholds and state-based decision making.
"""

import numpy as np
import pandas as pd
from channelpy import State, ChannelPipeline, ThresholdEncoder
from channelpy.adaptive import StreamingAdaptiveThreshold
from channelpy.visualization import plot_states, plot_threshold_adaptation


class TradingChannelSystem:
    """
    Complete trading system using channel algebra
    """
    
    def __init__(self):
        self.price_threshold = StreamingAdaptiveThreshold(window_size=500)
        self.volume_threshold = StreamingAdaptiveThreshold(window_size=500)
        self.pipeline = ChannelPipeline()
        
        # Add encoders
        self.pipeline.add_encoder(ThresholdEncoder(threshold_i=0.5, threshold_q=0.75))
        
        # Add interpreter
        self.pipeline.add_interpreter(self._interpret_states)
    
    def fit(self, prices: pd.Series, volumes: pd.Series):
        """Initialize with historical data"""
        for price, volume in zip(prices, volumes):
            self.price_threshold.update(price)
            self.volume_threshold.update(volume)
        return self
    
    def process_tick(self, price: float, volume: float) -> dict:
        """Process new market tick"""
        # Update thresholds
        self.price_threshold.update(price)
        self.volume_threshold.update(volume)
        
        # Encode
        price_state = self.price_threshold.encode(price)
        volume_state = self.volume_threshold.encode(volume)
        
        # Interpret
        return self._interpret_states(price_state, volume_state)
    
    def _interpret_states(self, price_state, volume_state):
        """Trading rules based on channel states"""
        from channelpy.core.state import PSI, DELTA, PHI, EMPTY
        
        if price_state == PSI and volume_state == PSI:
            return {'action': 'BUY', 'confidence': 1.0, 'reason': 'Strong momentum'}
        elif price_state == PSI and volume_state in [DELTA, PHI]:
            return {'action': 'BUY', 'confidence': 0.7, 'reason': 'Price momentum, weak volume'}
        elif price_state == EMPTY and volume_state == PSI:
            return {'action': 'SELL', 'confidence': 1.0, 'reason': 'Volume without price'}
        elif price_state == PHI and volume_state == PSI:
            return {'action': 'HOLD', 'confidence': 0.8, 'reason': 'Expected price, strong volume'}
        else:
            return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'No clear signal'}


def generate_sample_data(n_points=1000):
    """Generate sample price and volume data"""
    np.random.seed(42)
    
    # Generate price data with trend and noise
    trend = np.linspace(100, 120, n_points)
    noise = np.random.normal(0, 2, n_points)
    prices = trend + noise
    
    # Generate volume data with some correlation to price changes
    price_changes = np.diff(prices, prepend=prices[0])
    volume_base = np.random.exponential(1000, n_points)
    volume_spike = np.abs(price_changes) * 100
    volumes = volume_base + volume_spike
    
    return pd.Series(prices), pd.Series(volumes)


def main():
    """Main example function"""
    print("ChannelPy Trading System Example")
    print("=" * 40)
    
    # Generate sample data
    prices, volumes = generate_sample_data(1000)
    
    # Create trading system
    system = TradingChannelSystem()
    
    # Fit on first half of data
    split_point = len(prices) // 2
    system.fit(prices[:split_point], volumes[:split_point])
    
    print(f"Fitted on {split_point} historical points")
    print(f"Price threshold: {system.price_threshold.get_stats()}")
    print(f"Volume threshold: {system.volume_threshold.get_stats()}")
    print()
    
    # Process remaining data
    signals = []
    states = []
    price_thresholds = []
    volume_thresholds = []
    
    for i in range(split_point, len(prices)):
        price = prices.iloc[i]
        volume = volumes.iloc[i]
        
        signal = system.process_tick(price, volume)
        signals.append(signal)
        
        # Track states and thresholds
        price_state = system.price_threshold.encode(price)
        volume_state = system.volume_threshold.encode(volume)
        states.append((price_state, volume_state))
        
        price_thresholds.append(system.price_threshold.threshold_i)
        volume_thresholds.append(system.volume_threshold.threshold_q)
    
    # Analyze results
    actions = [s['action'] for s in signals]
    buy_signals = actions.count('BUY')
    sell_signals = actions.count('SELL')
    hold_signals = actions.count('HOLD')
    
    print(f"Trading Signals Generated:")
    print(f"  BUY:  {buy_signals}")
    print(f"  SELL: {sell_signals}")
    print(f"  HOLD: {hold_signals}")
    print()
    
    # Show some example signals
    print("Example Signals:")
    for i in range(0, min(10, len(signals))):
        signal = signals[i]
        price_state, volume_state = states[i]
        print(f"  {signal['action']:4} | {str(price_state):1}{str(volume_state):1} | {signal['reason']}")
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Plot price with adaptive thresholds
    fig1, ax1 = plot_threshold_adaptation(
        prices[split_point:].values,
        price_thresholds,
        volume_thresholds,
        title="Price with Adaptive Thresholds"
    )
    fig1.savefig('trading_thresholds.png', dpi=150, bbox_inches='tight')
    print("Saved: trading_thresholds.png")
    
    # Plot state sequence
    state_ints = [s[0].to_int() for s in states]
    state_array = StateArray.from_bits(
        i=[s[0].i for s in states],
        q=[s[0].q for s in states]
    )
    
    fig2, ax2 = plot_states(state_array, title="Price States Over Time")
    fig2.savefig('trading_states.png', dpi=150, bbox_inches='tight')
    print("Saved: trading_states.png")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()







