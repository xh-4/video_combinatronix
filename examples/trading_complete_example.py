"""
Complete Trading System Example

This example demonstrates a full trading system using ChannelPy:
1. Data generation and preprocessing
2. Technical indicator calculation
3. Channel encoding and state generation
4. Strategy implementation and backtesting
5. Performance analysis and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from channelpy.applications.trading import (
    TechnicalIndicators, TradingSignalEncoder, TradingChannelSystem,
    SimpleChannelStrategy, AdaptiveMomentumStrategy
)
from channelpy.examples.datasets import make_trading_data
from channelpy import plot_states, plot_state_distribution


def generate_trading_data():
    """Generate comprehensive trading dataset"""
    print("=== Generating Trading Data ===")
    
    # Generate base OHLCV data
    df = make_trading_data(
        n_samples=1000, 
        volatility=0.02, 
        drift=0.0005, 
        random_state=42
    )
    
    print(f"Generated {len(df)} trading days")
    print(f"Price range: [{df['close'].min():.2f}, {df['close'].max():.2f}]")
    print(f"Volume range: [{df['volume'].min():.0f}, {df['volume'].max():.0f}]")
    
    return df


def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    print("\n=== Calculating Technical Indicators ===")
    
    indicators = TechnicalIndicators()
    
    # Moving averages
    df['sma_20'] = indicators.sma(df['close'], 20)
    df['ema_12'] = indicators.ema(df['close'], 12)
    df['ema_26'] = indicators.ema(df['close'], 26)
    
    # RSI
    df['rsi'] = indicators.rsi(df['close'], 14)
    
    # MACD
    df['macd'], df['macd_signal'] = indicators.macd(df['close'])
    
    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = indicators.bollinger_bands(df['close'])
    
    # ATR
    df['atr'] = indicators.atr(df['high'], df['low'], df['close'])
    
    # OBV
    df['obv'] = indicators.obv(df['close'], df['volume'])
    
    print("Calculated indicators:")
    print(f"  SMA(20): {df['sma_20'].dropna().iloc[0]:.2f}")
    print(f"  RSI: {df['rsi'].dropna().iloc[0]:.2f}")
    print(f"  MACD: {df['macd'].dropna().iloc[0]:.2f}")
    print(f"  ATR: {df['atr'].dropna().iloc[0]:.2f}")
    
    return df


def demonstrate_signal_encoding(df):
    """Demonstrate signal encoding into channel states"""
    print("\n=== Signal Encoding ===")
    
    # Create encoder
    encoder = TradingSignalEncoder()
    encoder.fit(df)
    
    # Encode all channels
    channels = encoder.encode_all_channels(df)
    
    print("Channel encoding results:")
    for name, states in channels.items():
        print(f"  {name}: {states.count_by_state()}")
    
    # Show some examples
    print("\nFirst 10 price channel states:")
    price_states = channels['price']
    for i in range(10):
        print(f"  Day {i}: {price_states[i]}")
    
    return channels


def demonstrate_strategies(channels):
    """Demonstrate different trading strategies"""
    print("\n=== Trading Strategies ===")
    
    # Simple strategy
    print("1. Simple Channel Strategy:")
    simple_strategy = SimpleChannelStrategy()
    
    # Test with sample states
    sample_channels = {
        'price': channels['price'][0],
        'volume': channels['volume'][0],
        'volatility': channels['volatility'][0]
    }
    
    signal = simple_strategy.generate_signal(sample_channels)
    print(f"   Sample signal: {signal}")
    
    # Adaptive strategy
    print("\n2. Adaptive Momentum Strategy:")
    adaptive_strategy = AdaptiveMomentumStrategy()
    
    # Test FSM transitions
    print(f"   Initial mode: {adaptive_strategy.fsm.get_mode()}")
    
    # Process a sequence of states
    test_states = [channels['price'][i] for i in range(5)]
    for i, state in enumerate(test_states):
        signal = adaptive_strategy.generate_signal({'price': state})
        print(f"   Step {i}: {state} -> {signal} (mode: {adaptive_strategy.fsm.get_mode()})")


def run_backtest(df, strategy_name='simple'):
    """Run complete backtest"""
    print(f"\n=== Backtest: {strategy_name.upper()} Strategy ===")
    
    # Split data
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Training period: {len(train_df)} days")
    print(f"Testing period: {len(test_df)} days")
    
    # Create and fit system
    system = TradingChannelSystem(strategy=strategy_name)
    system.fit(train_df)
    
    # Run backtest
    results = system.backtest(test_df, initial_capital=10000.0, commission=0.001)
    
    # Display results
    print(f"\nBacktest Results:")
    print(f"  Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"  Final Capital: ${results['final_capital']:,.2f}")
    print(f"  Final Shares: {results['final_shares']:.2f}")
    print(f"  Final Equity: ${results['final_equity']:,.2f}")
    print(f"  Total Return: {results['total_return']:.2%}")
    print(f"  Number of Trades: {results['num_trades']}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
    
    return results


def compare_strategies(df):
    """Compare different strategies"""
    print("\n=== Strategy Comparison ===")
    
    strategies = ['simple', 'adaptive']
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy...")
        results[strategy] = run_backtest(df, strategy)
    
    # Compare results
    print("\n=== Strategy Comparison Results ===")
    print(f"{'Strategy':<12} {'Return':<10} {'Sharpe':<8} {'Trades':<8} {'Drawdown':<10}")
    print("-" * 60)
    
    for strategy, result in results.items():
        print(f"{strategy:<12} {result['total_return']:>8.2%} {result['sharpe_ratio']:>6.2f} "
              f"{result['num_trades']:>6} {result['max_drawdown']:>8.2%}")
    
    return results


def visualize_results(df, results):
    """Visualize trading results"""
    print("\n=== Creating Visualizations ===")
    
    # Plot 1: Price and Equity Curves
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Price curve
    ax1.plot(df.index, df['close'], label='Price', alpha=0.7)
    ax1.set_title('Price Movement')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Equity curves for both strategies
    for strategy, result in results.items():
        ax2.plot(result['equity_curve'], label=f'{strategy.title()} Strategy')
    
    ax2.set_title('Equity Curves')
    ax2.set_ylabel('Equity ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trading_results.png', dpi=150, bbox_inches='tight')
    print("Saved: trading_results.png")
    
    # Plot 2: State Distribution
    encoder = TradingSignalEncoder()
    encoder.fit(df)
    channels = encoder.encode_all_channels(df)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (name, states) in enumerate(channels.items()):
        plot_state_distribution(states, title=f'{name.title()} Channel', ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('trading_states.png', dpi=150, bbox_inches='tight')
    print("Saved: trading_states.png")


def analyze_trades(results):
    """Analyze individual trades"""
    print("\n=== Trade Analysis ===")
    
    for strategy, result in results.items():
        trades = result['trades']
        if not trades:
            print(f"{strategy.title()} strategy: No trades executed")
            continue
        
        print(f"\n{strategy.title()} Strategy Trades:")
        print(f"  Total trades: {len(trades)}")
        
        # Analyze buy vs sell trades
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        
        print(f"  Buy trades: {len(buy_trades)}")
        print(f"  Sell trades: {len(sell_trades)}")
        
        if buy_trades:
            avg_buy_price = np.mean([t['price'] for t in buy_trades])
            print(f"  Average buy price: ${avg_buy_price:.2f}")
        
        if sell_trades:
            avg_sell_price = np.mean([t['price'] for t in sell_trades])
            print(f"  Average sell price: ${avg_sell_price:.2f}")
        
        # Show first few trades
        print(f"  First 5 trades:")
        for i, trade in enumerate(trades[:5]):
            print(f"    {i+1}. {trade['action']} {trade['shares']:.2f} shares at ${trade['price']:.2f}")


def demonstrate_live_trading():
    """Demonstrate live trading system"""
    print("\n=== Live Trading System ===")
    
    from channelpy.applications.trading import LiveTradingSystem
    
    # Create live system
    live_system = LiveTradingSystem(strategy='adaptive', risk_per_trade=0.02)
    
    # Simulate live data stream
    print("Simulating live trading...")
    
    # Generate some live bars
    np.random.seed(42)
    base_price = 100.0
    current_capital = 10000.0
    
    for i in range(10):
        # Simulate live bar
        price_change = np.random.randn() * 0.02
        base_price *= (1 + price_change)
        
        bar = {
            'open': base_price,
            'high': base_price * 1.01,
            'low': base_price * 0.99,
            'close': base_price,
            'volume': np.random.randint(1000, 5000)
        }
        
        # Process bar
        signal = live_system.process_bar_live(bar, current_capital)
        
        print(f"  Bar {i+1}: Price=${bar['close']:.2f}, Signal={signal}")
        
        # Update capital (simplified)
        if signal['action'] == 'BUY':
            current_capital -= signal['size'] * bar['close']
        elif signal['action'] == 'SELL':
            current_capital += signal['size'] * bar['close']


def main():
    """Main trading example function"""
    print("ChannelPy Complete Trading System Example")
    print("=" * 50)
    
    # 1. Generate data
    df = generate_trading_data()
    
    # 2. Calculate indicators
    df = calculate_technical_indicators(df)
    
    # 3. Demonstrate encoding
    channels = demonstrate_signal_encoding(df)
    
    # 4. Demonstrate strategies
    demonstrate_strategies(channels)
    
    # 5. Run backtests
    results = compare_strategies(df)
    
    # 6. Visualize results
    visualize_results(df, results)
    
    # 7. Analyze trades
    analyze_trades(results)
    
    # 8. Demonstrate live trading
    demonstrate_live_trading()
    
    print("\n" + "=" * 50)
    print("Trading example completed successfully!")
    print("\nGenerated files:")
    print("- trading_results.png: Price and equity curves")
    print("- trading_states.png: State distributions")
    print("\nNext steps:")
    print("- Experiment with different strategy parameters")
    print("- Try different technical indicators")
    print("- Test with real market data")
    print("- Implement custom strategies")


if __name__ == "__main__":
    main()







