"""LLM-callable tools for managing worker needs.

Workers use these tools to check their needs and make purchasing decisions.
"""

import json
from typing import List, Dict


class NeedsTools:
    """Tools for workers to check and satisfy their needs."""

    def __init__(self, worker):
        """Initialize with a worker reference."""
        self.worker = worker

    def check_my_needs(self) -> str:
        """View current physiological needs and urgency.

        Returns:
            JSON string with needs state and currency available
        """
        needs = self.worker.needs
        return json.dumps({
            "hunger": needs.hunger,
            "thirst": needs.thirst,
            "rest": needs.rest,
            "recreation": needs.recreation,
            "housing_satisfaction": needs.housing_satisfaction,
            "critical_needs": needs.get_critical_needs(),
            "summary": needs.get_needs_summary(),
            "currency_available": self.worker.currency
        }, indent=2)

    def view_available_goods(self) -> str:
        """View consumables and services available for purchase.

        Returns:
            JSON string with available consumables and services
        """
        pod = self.worker.pod
        if not pod:
            return json.dumps({"error": "Not in a pod"}, indent=2)

        # Gather available consumables from pod inventory
        available_consumables = {}
        if hasattr(pod.inventory, 'consumables'):
            for name, consumable in pod.inventory.consumables.items():
                if consumable.quantity > 0:
                    available_consumables[name] = {
                        "cost": consumable.info.base_value,
                        "need_type": consumable.need_type,
                        "satisfaction": consumable.satisfaction_value,
                        "available": consumable.quantity
                    }

        # Add abstracted services/purchases
        available = {
            "consumables": available_consumables,
            "services": {
                "rest": {
                    "cost": 0.0,
                    "satisfaction": 0.6,
                    "available": True,
                    "description": "Take time to rest and recover"
                },
                "home_furnishings": {
                    "cost": 20.0,
                    "satisfaction": 0.2,
                    "available": True,
                    "description": "Improve your living conditions (abstracted)"
                }
            }
        }
        return json.dumps(available, indent=2)

    def purchase_and_consume(self, item_type: str, item_name: str) -> str:
        """Purchase and consume an item or service to satisfy needs.

        Args:
            item_type: "consumable" or "service"
            item_name: Name of the item/service to purchase

        Returns:
            JSON string with result and updated need levels
        """
        pod = self.worker.pod
        if not pod:
            return json.dumps({"success": False, "error": "Not in a pod"}, indent=2)

        if item_type == "consumable":
            # Check if consumable exists in new system
            if not hasattr(pod.inventory, 'consumables'):
                return json.dumps({
                    "success": False,
                    "error": "Consumables not available in this pod"
                }, indent=2)

            consumable = pod.inventory.consumables.get(item_name)
            if not consumable or consumable.quantity < 1.0:
                return json.dumps({
                    "success": False,
                    "error": f"{item_name} not available"
                }, indent=2)

            # Check currency
            if self.worker.currency < consumable.info.base_value:
                return json.dumps({
                    "success": False,
                    "error": "Insufficient funds"
                }, indent=2)

            # Purchase and consume
            self.worker.currency -= consumable.info.base_value
            satisfaction = consumable.consume()

            # Satisfy appropriate need
            if consumable.need_type == "hunger":
                self.worker.needs.consume_food(satisfaction)
            elif consumable.need_type == "thirst":
                self.worker.needs.consume_water(satisfaction)
            elif consumable.need_type == "recreation":
                self.worker.needs.enjoy_recreation(satisfaction)

            # Record purchase
            self.worker.needs.record_purchase(item_name, consumable.info.base_value, consumable.need_type)

            return json.dumps({
                "success": True,
                "purchased": item_name,
                "cost": consumable.info.base_value,
                "need_satisfied": consumable.need_type,
                "new_currency": self.worker.currency,
                "new_need_level": getattr(self.worker.needs, consumable.need_type)
            }, indent=2)

        elif item_type == "service":
            if item_name == "rest":
                self.worker.needs.sleep()
                return json.dumps({
                    "success": True,
                    "rested": True,
                    "cost": 0.0,
                    "new_rest_level": self.worker.needs.rest
                }, indent=2)

            elif item_name == "home_furnishings":
                cost = 20.0
                if self.worker.currency < cost:
                    return json.dumps({
                        "success": False,
                        "error": "Insufficient funds"
                    }, indent=2)

                self.worker.currency -= cost
                self.worker.needs.improve_housing(0.2)
                self.worker.needs.record_purchase("home_furnishings", cost, "housing")

                return json.dumps({
                    "success": True,
                    "purchased": "home improvements",
                    "cost": cost,
                    "new_currency": self.worker.currency,
                    "new_housing_satisfaction": self.worker.needs.housing_satisfaction
                }, indent=2)

        return json.dumps({
            "success": False,
            "error": "Invalid item type"
        }, indent=2)


def get_needs_tool_definitions() -> List[Dict]:
    """Return OpenAI function calling definitions for needs tools.

    Returns:
        List of tool definitions in OpenAI format
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "check_my_needs",
                "description": "Check your current hunger, thirst, rest, recreation, and housing satisfaction levels",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "view_available_goods",
                "description": "View food, water, recreation items, and services available for purchase from your pod",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "purchase_and_consume",
                "description": "Purchase and consume food/water/recreation or pay for services like rest or home improvements to satisfy needs",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item_type": {
                            "type": "string",
                            "enum": ["consumable", "service"],
                            "description": "Type of purchase"
                        },
                        "item_name": {
                            "type": "string",
                            "description": "Name of item to purchase (e.g., 'bread', 'water', 'rest', 'home_furnishings')"
                        }
                    },
                    "required": ["item_type", "item_name"]
                }
            }
        }
    ]
