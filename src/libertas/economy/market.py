# src/libertas/economy/market.py
"""Market system for external trade with stochastic prices and shocks."""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from random import Random


@dataclass
class MarketOrder:
    """An order to buy or sell in the market."""
    order_id: str
    worker_id: str
    pod_id: str
    resource_name: str
    quantity: float
    price_limit: float  # Max price for buy, min price for sell
    order_type: str  # "buy" or "sell"
    timestamp: int
    is_active: bool = True
    filled_quantity: float = 0.0
    average_price: float = 0.0

@dataclass
class MarketPrice:
    """Current price information for a resource."""
    current_price: float
    base_price: float
    volatility: float
    trend: float = 0.0
    last_update: int = 0
    price_history: List[float] = field(default_factory=list)
    
    def add_history(self, price: float) -> None:
        """Add price to history, keeping last 100 prices."""
        self.price_history.append(price)
        if len(self.price_history) > 100:
            self.price_history.pop(0)


class Market:
    """
    External market for buying and selling resources.
    
    Features:
    - Stochastic price modeling with mean reversion
    - Supply/demand pressure from orders
    - Random market shocks
    - Price history tracking
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        self.random = Random(random_seed)
        self.prices: Dict[str, MarketPrice] = {}
        self.orders: List[MarketOrder] = []
        self.order_counter = 0
        self.shock_history: List[Dict] = []
        
        # Market parameters
        self.volatility_base = 0.1  # Base price volatility
        self.mean_reversion_rate = 0.05  # How fast prices revert to base
        self.shock_probability = 0.01  # Chance of shock per step
        self.shock_magnitude = 0.5  # Max shock magnitude (50% change)
    
    def register_resource(self, resource_name: str, base_price: float, 
                         volatility: Optional[float] = None) -> None:
        """Register a resource for trading on the market."""
        if resource_name not in self.prices:
            self.prices[resource_name] = MarketPrice(
                current_price=base_price,
                base_price=base_price,
                volatility=volatility or self.volatility_base,
                price_history=[base_price]
            )
    
    def get_current_price(self, resource_name: str) -> float:
        """Get current market price for a resource."""
        if resource_name not in self.prices:
            return 0.0
        return self.prices[resource_name].current_price
    
    def get_price_history(self, resource_name: str) -> List[float]:
        """Get price history for a resource."""
        if resource_name not in self.prices:
            return []
        return self.prices[resource_name].price_history.copy()
    
    def place_order(self, worker_id: str, pod_id: str, resource_name: str,
                   quantity: float, price_limit: float, order_type: str,
                   timestamp: int) -> MarketOrder:
        """Place a buy or sell order."""
        self.order_counter += 1
        order = MarketOrder(
            order_id=f"order_{self.order_counter}",
            worker_id=worker_id,
            pod_id=pod_id,
            resource_name=resource_name,
            quantity=quantity,
            price_limit=price_limit,
            order_type=order_type,
            timestamp=timestamp
        )
        self.orders.append(order)
        return order
    
    def process_market(self, timestamp: int, pod_inventory_getter, 
                  pod_inventory_setter) -> List[Dict]:
        """
        Process all active market orders.
        
        Args:
            timestamp: Current simulation step
            pod_inventory_getter: Function to get pod inventory
            pod_inventory_setter: Function to update pod inventory
        
        Returns:
            List of transaction records
        """
        transactions = []
        
        # First, apply price updates from supply/demand
        self._update_prices(timestamp)
        
        # Separate buy and sell orders
        buy_orders = [o for o in self.orders if o.is_active and o.order_type == "buy"]
        sell_orders = [o for o in self.orders if o.is_active and o.order_type == "sell"]
        
        # Sort by price (highest buy first, lowest sell first)
        buy_orders.sort(key=lambda o: o.price_limit, reverse=True)
        sell_orders.sort(key=lambda o: o.price_limit)
        
        # Match orders
        for buy_order in buy_orders[:]:
            for sell_order in sell_orders[:]:
                if buy_order.resource_name != sell_order.resource_name:
                    continue
                
                # Check if resource is registered
                if buy_order.resource_name not in self.prices:
                    continue
                
                # Check if prices match (buy limit >= sell limit)
                if buy_order.price_limit >= sell_order.price_limit:
                    # Check seller's actual inventory
                    seller_inventory = pod_inventory_getter(sell_order.pod_id, buy_order.resource_name)
                    available_quantity = min(
                        seller_inventory,
                        sell_order.quantity - sell_order.filled_quantity
                    )
                    
                    # Calculate trade quantity
                    trade_quantity = min(
                        buy_order.quantity - buy_order.filled_quantity,
                        available_quantity
                    )
                    
                    if trade_quantity > 0:
                        # Execute trade at average of limits
                        trade_price = (buy_order.price_limit + sell_order.price_limit) / 2
                        
                        # Transfer resources
                        pod_inventory_setter(
                            sell_order.pod_id, 
                            buy_order.resource_name, 
                            -trade_quantity  # Remove from seller
                        )
                        pod_inventory_setter(
                            buy_order.pod_id,
                            buy_order.resource_name,
                            trade_quantity  # Add to buyer
                        )
                        
                        # Record transaction
                        transaction = {
                            'order_id': f"tx_{timestamp}_{len(transactions)}",
                            'resource': buy_order.resource_name,
                            'quantity': trade_quantity,
                            'price': trade_price,
                            'total_value': trade_quantity * trade_price,
                            'buyer_pod': buy_order.pod_id,
                            'buyer_worker': buy_order.worker_id,
                            'seller_pod': sell_order.pod_id,
                            'seller_worker': sell_order.worker_id,
                            'timestamp': timestamp
                        }
                        transactions.append(transaction)
                        
                        # Update order fill quantities
                        buy_order.filled_quantity += trade_quantity
                        sell_order.filled_quantity += trade_quantity
                        
                        # Calculate average prices
                        if buy_order.filled_quantity > 0:
                            buy_order.average_price = (
                                (buy_order.average_price * (buy_order.filled_quantity - trade_quantity) +
                                trade_price * trade_quantity) / buy_order.filled_quantity
                            )
                        if sell_order.filled_quantity > 0:
                            sell_order.average_price = (
                                (sell_order.average_price * (sell_order.filled_quantity - trade_quantity) +
                                trade_price * trade_quantity) / sell_order.filled_quantity
                            )
        
        # Deactivate fully filled orders
        for order in self.orders:
            if order.is_active and order.filled_quantity >= order.quantity:
                order.is_active = False
        
        # Clean up inactive orders
        self.orders = [o for o in self.orders if o.is_active]
        
        return transactions
    
    def _update_prices(self, timestamp: int):
        """Update market prices with random walk and mean reversion."""
        for resource_name, price_info in self.prices.items():
            # Random walk component
            shock = self.random.gauss(0, price_info.volatility)
            
            # Mean reversion component
            reversion = self.mean_reversion_rate * (price_info.base_price - price_info.current_price)
            
            # Update price
            price_info.current_price *= (1 + shock + reversion)
            price_info.current_price = max(0.1, price_info.current_price)  # No negative prices
            
            # Apply random market shock
            if self.random.random() < self.shock_probability:
                shock_direction = 1 if self.random.random() > 0.5 else -1
                shock_magnitude = self.random.uniform(0, self.shock_magnitude)
                price_info.current_price *= (1 + shock_direction * shock_magnitude)
                
                self.shock_history.append({
                    'resource': resource_name,
                    'old_price': price_info.price_history[-1] if price_info.price_history else price_info.current_price,
                    'new_price': price_info.current_price,
                    'magnitude': shock_magnitude,
                    'timestamp': timestamp
                })
            
            price_info.add_history(price_info.current_price)
            price_info.last_update = timestamp
    
    def apply_external_shock(self, resource_name: str, factor: float) -> None:
        """Apply a manual shock to a resource price."""
        if resource_name in self.prices:
            old_price = self.prices[resource_name].current_price
            self.prices[resource_name].current_price *= factor
            self.shock_history.append({
                'resource': resource_name,
                'old_price': old_price,
                'new_price': self.prices[resource_name].current_price,
                'magnitude': abs(factor - 1),
                'timestamp': 'manual',
                'manual': True
            })
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get summary of current market state."""
        return {
            'prices': {
                name: {
                    'current': price.current_price,
                    'base': price.base_price,
                    'change': ((price.current_price - price.base_price) / price.base_price * 100) if price.base_price > 0 else 0,
                    'trend': price.trend
                }
                for name, price in self.prices.items()
            },
            'active_orders': len([o for o in self.orders if o.is_active]),
            'total_shocks': len(self.shock_history),
            'recent_shocks': self.shock_history[-5:] if self.shock_history else []
        }
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        for order in self.orders:
            if order.order_id == order_id and order.is_active:
                order.is_active = False
                return True
        return False
    
    def get_worker_orders(self, worker_id: str) -> List[MarketOrder]:
        """Get all active orders for a worker."""
        return [o for o in self.orders if o.is_active and o.worker_id == worker_id]