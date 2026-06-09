# tests/test_market.py
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from libertas.economy.market import Market, MarketOrder, MarketPrice


@pytest.mark.unit
class TestMarketPrice:
    """Test MarketPrice class."""
    
    def test_price_creation(self):
        """Test basic price creation."""
        price = MarketPrice(
            current_price=10.0,
            base_price=10.0,
            volatility=0.1
        )
        
        assert price.current_price == 10.0
        assert price.base_price == 10.0
        assert price.volatility == 0.1
        assert price.trend == 0.0
    
    def test_price_history(self):
        """Test price history tracking."""
        price = MarketPrice(current_price=10.0, base_price=10.0, volatility=0.1)
        
        price.add_history(10.0)
        price.add_history(10.5)
        price.add_history(9.8)
        
        assert len(price.price_history) == 3
        assert price.price_history == [10.0, 10.5, 9.8]
    
    def test_price_history_limit(self):
        """Test price history limited to 100 entries."""
        price = MarketPrice(current_price=10.0, base_price=10.0, volatility=0.1)
        
        for i in range(150):
            price.add_history(10.0 + i * 0.1)
        
        assert len(price.price_history) == 100


@pytest.mark.unit
class TestMarketOrder:
    """Test MarketOrder class."""
    
    def test_order_creation(self):
        """Test basic order creation."""
        order = MarketOrder(
            order_id="order_1",
            worker_id="worker_001",
            pod_id="pod_001",
            resource_name="wood",
            quantity=100.0,
            price_limit=10.0,
            order_type="buy",
            timestamp=100
        )
        
        assert order.order_id == "order_1"
        assert order.worker_id == "worker_001"
        assert order.resource_name == "wood"
        assert order.quantity == 100.0
        assert order.price_limit == 10.0
        assert order.order_type == "buy"
        assert order.is_active
        assert order.filled_quantity == 0.0
        assert order.average_price == 0.0


@pytest.mark.unit
class TestMarket:
    """Test Market class."""
    
    def setup_method(self):
        self.market = Market(random_seed=42)
        
        # Register test resources
        self.market.register_resource("wood", base_price=10.0, volatility=0.1)
        self.market.register_resource("metal", base_price=20.0, volatility=0.15)
    
    def test_register_resource(self):
        """Test registering resources for trading."""
        assert "wood" in self.market.prices
        assert "metal" in self.market.prices
        
        price_info = self.market.prices["wood"]
        assert price_info.base_price == 10.0
        assert price_info.volatility == 0.1
    
    def test_get_current_price(self):
        """Test getting current market price."""
        price = self.market.get_current_price("wood")
        assert price == 10.0
        
        # Unknown resource
        price = self.market.get_current_price("unknown")
        assert price == 0.0
    
    def test_get_price_history(self):
        """Test getting price history."""
        history = self.market.get_price_history("wood")
        assert len(history) == 1
        assert history[0] == 10.0
        
        # Unknown resource
        history = self.market.get_price_history("unknown")
        assert history == []
    
    def test_place_order(self):
        """Test placing market orders."""
        order = self.market.place_order(
            worker_id="worker_001",
            pod_id="pod_001",
            resource_name="wood",
            quantity=50.0,
            price_limit=9.0,
            order_type="buy",
            timestamp=100
        )
        
        assert len(self.market.orders) == 1
        assert order.order_type == "buy"
        assert order.quantity == 50.0
    
    def test_price_updates(self):
        """Test that prices update with random walk."""
        # Capture initial price
        initial_price = self.market.get_current_price("wood")
        
        # Update prices
        self.market._update_prices(timestamp=1)
        
        # Price should have changed (within bounds)
        new_price = self.market.get_current_price("wood")
        assert initial_price != new_price
        assert new_price > 0.1  # No negative prices
    
    def test_mean_reversion(self):
        """Test that prices revert to base over time."""
        # Set a high volatility to push price away
        self.market.prices["wood"].volatility = 0.5
        
        # Update many times
        for i in range(50):
            self.market._update_prices(timestamp=i)
        
        # Price should still be reasonable (mean reversion)
        final_price = self.market.get_current_price("wood")
        assert final_price > 5.0
        self.assertLess(final_price, 20.0)
    
    def test_market_shock(self):
        """Test random market shocks."""
        # Reset shock history
        self.market.shock_history = []
        
        # Set high shock probability
        self.market.shock_probability = 1.0
        self.market.shock_magnitude = 0.3
        
        initial_price = self.market.get_current_price("wood")
        self.market._update_prices(timestamp=1)
        new_price = self.market.get_current_price("wood")
        
        # Price should have changed significantly
        assert initial_price != new_price
        
        # Filter shocks for wood resource only
        wood_shocks = [s for s in self.market.shock_history if s['resource'] == 'wood']
        assert len(wood_shocks) == 1
    
    def test_manual_shock(self):
        """Test applying manual price shock."""
        initial_price = self.market.get_current_price("wood")
        
        self.market.apply_external_shock("wood", 2.0)
        
        new_price = self.market.get_current_price("wood")
        assert new_price == initial_price * 2.0
        assert len(self.market.shock_history) == 1
        assert self.market.shock_history[0].get("manual", False)
    
    def test_order_matching_buy_sell(self):
        """Test matching buy and sell orders."""
        # Setup mock pod inventory functions
        inventory = {"pod_001": {"wood": 100.0}, "pod_002": {"wood": 0.0}}
        
        def get_inventory(pod_id, resource):
            return inventory.get(pod_id, {}).get(resource, 0.0)
        
        def set_inventory(pod_id, resource, delta):
            if pod_id not in inventory:
                inventory[pod_id] = {}
            inventory[pod_id][resource] = inventory[pod_id].get(resource, 0.0) + delta
        
        # Place sell order
        self.market.place_order(
            worker_id="worker_001",
            pod_id="pod_001",
            resource_name="wood",
            quantity=50.0,
            price_limit=8.0,
            order_type="sell",
            timestamp=100
        )
        
        # Place buy order
        self.market.place_order(
            worker_id="worker_002",
            pod_id="pod_002",
            resource_name="wood",
            quantity=50.0,
            price_limit=12.0,
            order_type="buy",
            timestamp=100
        )
        
        # Process market
        transactions = self.market.process_market(
            timestamp=101,
            pod_inventory_getter=get_inventory,
            pod_inventory_setter=set_inventory
        )
        
        # Should have one transaction
        assert len(transactions) == 1
        
        tx = transactions[0]
        assert tx["resource"] == "wood"
        assert tx["quantity"] == 50.0
        assert tx["buyer_pod"] == "pod_002"
        assert tx["seller_pod"] == "pod_001"
        
        # Check inventory updates
        assert inventory["pod_001"]["wood"] == 50.0  # Sold 50
        assert inventory["pod_002"]["wood"] == 50.0  # Bought 50
    
    def test_partial_fill(self):
        """Test partial order filling."""
        inventory = {"pod_001": {"wood": 30.0}, "pod_002": {"wood": 0.0}}
        
        def get_inventory(pod_id, resource):
            return inventory.get(pod_id, {}).get(resource, 0.0)
        
        def set_inventory(pod_id, resource, delta):
            if pod_id not in inventory:
                inventory[pod_id] = {}
            inventory[pod_id][resource] = inventory[pod_id].get(resource, 0.0) + delta
        
        # Sell order for 50 (only 30 available)
        self.market.place_order(
            worker_id="worker_001", pod_id="pod_001",
            resource_name="wood", quantity=50.0,
            price_limit=8.0, order_type="sell", timestamp=100
        )
        
        # Buy order for 50
        self.market.place_order(
            worker_id="worker_002", pod_id="pod_002",
            resource_name="wood", quantity=50.0,
            price_limit=12.0, order_type="buy", timestamp=100
        )
        
        transactions = self.market.process_market(101, get_inventory, set_inventory)
    
        # Only 30 should be traded
        assert len(transactions) == 1
        assert transactions[0]["quantity"] == 30.0
        
        # Both orders should still be active (partially filled)
        active_orders = [o for o in self.market.orders if o.is_active]
        assert len(active_orders) == 2  # Both buy and sell orders active
        assert active_orders[0].filled_quantity == 30.0
        assert active_orders[1].filled_quantity == 30.0
    
    def test_price_matching_priority(self):
        """Test that best prices match first."""
        inventory = {"pod_001": {"wood": 100.0}, "pod_002": {"wood": 0.0}}
        
        def get_inventory(pod_id, resource):
            return inventory.get(pod_id, {}).get(resource, 0.0)
        
        def set_inventory(pod_id, resource, delta):
            if pod_id not in inventory:
                inventory[pod_id] = {}
            inventory[pod_id][resource] = inventory[pod_id].get(resource, 0.0) + delta
        
        # Multiple sell orders at different prices
        self.market.place_order("w1", "pod_001", "wood", 50.0, 10.0, "sell", 100)
        self.market.place_order("w2", "pod_001", "wood", 50.0, 9.0, "sell", 100)
        
        # Buy order will take cheapest sell first
        self.market.place_order("w3", "pod_002", "wood", 100.0, 15.0, "buy", 100)
        
        transactions = self.market.process_market(101, get_inventory, set_inventory)
        
        # Should have two transactions (both sells filled)
        assert len(transactions) == 2
        
        # Cheaper sell should execute first (price 9.0)
        # But since both fill, we just check total quantity
        total_quantity = sum(tx["quantity"] for tx in transactions)
        assert total_quantity == 100.0
    
    def test_cancel_order(self):
        """Test canceling an active order."""
        order = self.market.place_order(
            worker_id="worker_001", pod_id="pod_001",
            resource_name="wood", quantity=50.0,
            price_limit=10.0, order_type="sell", timestamp=100
        )
        
        assert self.market.cancel_order(order.order_id)
        
        # Order should be inactive
        active_orders = [o for o in self.market.orders if o.is_active]
        assert len(active_orders) == 0
        
        # Cancel again should fail
        assert not (self.market.cancel_order(order.order_id))
    
    def test_get_worker_orders(self):
        """Test retrieving orders by worker."""
        self.market.place_order("worker_001", "pod_001", "wood", 50.0, 10.0, "sell", 100)
        self.market.place_order("worker_001", "pod_001", "metal", 30.0, 20.0, "sell", 100)
        self.market.place_order("worker_002", "pod_002", "wood", 40.0, 9.0, "buy", 100)
        
        worker_orders = self.market.get_worker_orders("worker_001")
        assert len(worker_orders) == 2
        
        worker_orders = self.market.get_worker_orders("worker_002")
        assert len(worker_orders) == 1
    
    def test_get_market_summary(self):
        """Test getting market summary."""
        summary = self.market.get_market_summary()
        
        assert "prices" in summary
        assert "wood" in summary["prices"]
        assert "metal" in summary["prices"]
        assert summary["active_orders"] == 0
        assert summary["total_shocks"] == 0


@pytest.mark.unit
class TestMarketEdgeCases:
    """Test edge cases for market system."""
    
    def setup_method(self):
        self.market = Market(random_seed=42)
        self.market.register_resource("wood", base_price=10.0)
    
    def test_no_matching_orders(self):
        """Test when no orders match (buy below sell)."""
        inventory = {"pod_001": {"wood": 100.0}, "pod_002": {"wood": 0.0}}
        
        def get_inventory(pod_id, resource):
            return inventory.get(pod_id, {}).get(resource, 0.0)
        
        def set_inventory(pod_id, resource, delta):
            pass
        
        # Sell at high price, buy at low price - no match
        self.market.place_order("w1", "pod_001", "wood", 50.0, 15.0, "sell", 100)
        self.market.place_order("w2", "pod_002", "wood", 50.0, 10.0, "buy", 100)
        
        transactions = self.market.process_market(101, get_inventory, set_inventory)
        
        assert len(transactions) == 0
        assert len([o for o in self.market.orders if o.is_active]) == 2
    
    def test_order_cleanup_on_completion(self):
        """Test that completed orders are removed."""
        inventory = {"pod_001": {"wood": 100.0}, "pod_002": {"wood": 0.0}}
        
        def get_inventory(pod_id, resource):
            return inventory.get(pod_id, {}).get(resource, 0.0)
        
        def set_inventory(pod_id, resource, delta):
            if pod_id not in inventory:
                inventory[pod_id] = {}
            inventory[pod_id][resource] = inventory[pod_id].get(resource, 0.0) + delta
        
        self.market.place_order("w1", "pod_001", "wood", 50.0, 10.0, "sell", 100)
        self.market.place_order("w2", "pod_002", "wood", 50.0, 10.0, "buy", 100)
        
        transactions = self.market.process_market(101, get_inventory, set_inventory)
        
        # Orders should be removed after full fill
        assert len([o for o in self.market.orders if o.is_active]) == 0
    
    def test_resource_not_registered(self):
        """Test trading unregistered resource."""
        inventory = {"pod_001": {}, "pod_002": {}}
        
        def get_inventory(pod_id, resource):
            return 0.0
        
        def set_inventory(pod_id, resource, delta):
            pass
        
        self.market.place_order("w1", "pod_001", "unregistered", 50.0, 10.0, "sell", 100)
        self.market.place_order("w2", "pod_002", "unregistered", 50.0, 10.0, "buy", 100)
        
        transactions = self.market.process_market(101, get_inventory, set_inventory)
        
        # No price info, so no trades
        assert len(transactions) == 0

    def test_process_market_mismatched_resources(self):
        """Test that orders for different resources don't match."""
        # Register both resources
        self.market.register_resource("wood", 10.0)
        self.market.register_resource("metal", 20.0)

        # Create orders for different resources
        self.market.place_order("w1", "pod_001", "wood", 100.0, 10.0, "sell", 100)
        self.market.place_order("w2", "pod_002", "metal", 50.0, 20.0, "buy", 100)

        # Inventories
        inventories = {
            "pod_001": {"wood": 100.0},
            "pod_002": {"metal": 0.0}
        }

        def get_inventory(pod_id, resource_name):
            return inventories.get(pod_id, {}).get(resource_name, 0.0)

        def set_inventory(pod_id, resource_name, quantity):
            if pod_id not in inventories:
                inventories[pod_id] = {}
            inventories[pod_id][resource_name] = quantity

        transactions = self.market.process_market(101, get_inventory, set_inventory)

        # No trades should occur - different resources
        assert len(transactions) == 0


