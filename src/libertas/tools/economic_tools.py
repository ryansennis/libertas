# src/libertas/tools/economic_tools.py
"""Function calling tools for workers to interact with the economy."""

from typing import Dict, List, Optional
import json


class EconomicTools:
    """
    Tool functions that workers can call to interact with the economy.
    
    These are registered with the LLM agent and called via function calling.
    """
    
    def __init__(self, worker):
        self.worker = worker
    
    def get_pod(self):
        """Get the pod this worker belongs to."""
        return self.worker.pod
    
    def get_federation(self):
        """Get the federation this worker belongs to."""
        return self.worker.federation
    
    # ===== Inventory & Information Tools =====
    
    def inspect_inventory(self) -> str:
        """
        Check the current inventory of your pod.
        
        Returns:
            JSON string with inventory contents
        """
        pod = self.get_pod()
        if not pod:
            return json.dumps({"error": "Worker not assigned to a pod"})
        
        inventory = pod.get_inventory_summary()
        tools = pod.get_tools_summary()
        
        return json.dumps({
            "pod_id": pod.name,
            "inventory": inventory,
            "tools": tools,
            "total_items": sum(inventory.values()),
            "total_tools": sum(tools.values())
        }, indent=2)
    
    def inspect_worker_status(self) -> str:
        """
        Check your own status as a worker.
        
        Returns:
            JSON string with worker status
        """
        status = self.worker.get_status()
        return json.dumps(status, indent=2)
    
    def list_known_resources(self) -> str:
        """
        List all known resource types in the federation.
        
        Returns:
            JSON string with resource list
        """
        federation = self.get_federation()
        resources = federation.list_resources()
        
        resource_details = []
        for name in resources:
            resource = federation.get_resource(name)
            if resource:
                resource_details.append({
                    "name": resource.name,
                    "is_tool": resource.is_tool,
                    "base_value": resource.base_value
                })
        
        return json.dumps({
            "count": len(resource_details),
            "resources": resource_details
        }, indent=2)
    
    def list_known_recipes(self) -> str:
        """
        List all known production recipes in the federation.
        
        Returns:
            JSON string with recipe list
        """
        federation = self.get_federation()
        recipes = federation.list_recipes()
        
        recipe_details = []
        for name in recipes:
            recipe = federation.get_recipe(name)
            if recipe:
                recipe_details.append({
                    "name": recipe.name,
                    "category": recipe.category,
                    "duration": recipe.total_duration,
                    "inputs": recipe.total_inputs,
                    "outputs": recipe.total_outputs,
                    "requires_tools": list(recipe.requires_tools),
                    "requires_skills": list(recipe.requires_skills)
                })
        
        return json.dumps({
            "count": len(recipe_details),
            "recipes": recipe_details
        }, indent=2)
    
    def get_recipe_details(self, recipe_name: str) -> str:
        """
        Get detailed information about a specific recipe.
        
        Args:
            recipe_name: Name of the recipe to inspect
        
        Returns:
            JSON string with recipe details
        """
        federation = self.get_federation()
        recipe = federation.get_recipe(recipe_name)
        
        if not recipe:
            return json.dumps({"error": f"Recipe '{recipe_name}' not found"})
        
        steps = []
        for i, step in enumerate(recipe.steps):
            steps.append({
                "index": i,
                "name": step.name,
                "type": step.step_type.value,
                "duration": step.duration,
                "inputs": step.inputs,
                "outputs": step.outputs,
                "required_tool": step.required_tool,
                "required_skill": step.required_skill,
                "required_skill_level": step.required_skill_level
            })
        
        return json.dumps({
            "name": recipe.name,
            "category": recipe.category,
            "description": recipe.description,
            "total_duration": recipe.total_duration,
            "steps": steps,
            "total_inputs": recipe.total_inputs,
            "total_outputs": recipe.total_outputs,
            "requires_tools": list(recipe.requires_tools),
            "requires_skills": list(recipe.requires_skills)
        }, indent=2)
    
    # ===== Production Tools =====
    
    def start_production(self, recipe_name: str, batch_size: int = 1) -> str:
        """
        Start producing a recipe in your pod.
        
        Args:
            recipe_name: Name of the recipe to produce
            batch_size: Number of batches to produce (default: 1)
        
        Returns:
            Status message with job ID if successful
        """
        pod = self.get_pod()
        if not pod:
            return json.dumps({"error": "Worker not assigned to a pod"})
        
        success, result = pod.start_production(
            recipe_name=recipe_name,
            batch_size=batch_size,
            started_by=self.worker.unique_id
        )
        
        if success:
            return json.dumps({
                "success": True,
                "job_id": result,
                "message": f"Started production of {batch_size}x {recipe_name}"
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "error": result
            }, indent=2)
    
    def check_production_queue(self) -> str:
        """
        Check the current production queue in your pod.
        
        Returns:
            JSON string with active and queued jobs
        """
        pod = self.get_pod()
        if not pod:
            return json.dumps({"error": "Worker not assigned to a pod"})
        
        active_jobs = []
        for job in pod.active_jobs:
            active_jobs.append({
                "job_id": job.job_id,
                "recipe": job.recipe.name,
                "progress": job.get_progress(),
                "current_step": job.current_step_index,
                "total_steps": len(job.recipe.steps),
                "assigned_worker": job.assigned_worker_id
            })
        
        queued_jobs = []
        for job in pod.production_queue:
            queued_jobs.append({
                "job_id": job.job_id,
                "recipe": job.recipe.name,
                "batch_size": job.batch_size
            })
        
        return json.dumps({
            "active_jobs": active_jobs,
            "queued_jobs": queued_jobs,
            "completed_count": len(pod.completed_jobs)
        }, indent=2)
    
    # ===== Tool Management Tools =====
    
    def list_my_tools(self) -> str:
        """
        List all tools in your personal inventory.
        
        Returns:
            JSON string with tool list
        """
        tools = []
        for tool_name, tool_list in self.worker.tools.items():
            for i, tool in enumerate(tool_list):
                tools.append({
                    "name": tool_name,
                    "instance_id": i,
                    "durability": tool.durability,
                    "required_skill": tool.required_skill
                })
        
        return json.dumps({
            "equipped_tool": self.worker.equipped_tool,
            "tools": tools,
            "total_tools": len(tools)
        }, indent=2)
    
    def equip_tool(self, tool_name: str) -> str:
        """
        Equip a tool from your inventory.
        
        Args:
            tool_name: Name of the tool to equip
        
        Returns:
            Status message
        """
        if self.worker.equip_tool(tool_name):
            return json.dumps({
                "success": True,
                "message": f"Equipped {tool_name}"
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "error": f"Cannot equip {tool_name}. Tool not found in inventory."
            }, indent=2)
    
    def unequip_tool(self) -> str:
        """
        Unequip your current tool.
        
        Returns:
            Status message
        """
        old_tool = self.worker.equipped_tool
        self.worker.unequip_tool()
        
        return json.dumps({
            "success": True,
            "message": f"Unequipped {old_tool}" if old_tool else "No tool was equipped"
        }, indent=2)
    
    # ===== Transfer Tools =====
    
    def transfer_to_pod(self, resource_name: str, quantity: float, target_pod_name: str) -> str:
        """
        Transfer resources from your pod to another pod.

        Args:
            resource_name: Name of the resource to transfer
            quantity: Amount to transfer
            target_pod_name: Name of the target pod

        Returns:
            Status message
        """
        pod = self.get_pod()
        if not pod:
            return json.dumps({"error": "Worker not assigned to a pod"})

        federation = self.get_federation()
        target_pod = federation.get_pod_by_name(target_pod_name)
        
        if not target_pod:
            return json.dumps({"error": f"Pod '{target_pod_name}' not found"})

        if pod.transfer_to_pod(resource_name, quantity, target_pod):
            return json.dumps({
                "success": True,
                "message": f"Transferred {quantity} {resource_name} from {pod.name} to {target_pod_name}"
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "error": f"Failed to transfer {quantity} {resource_name}. Insufficient inventory."
            }, indent=2)
    
    def list_pods(self) -> str:
        """
        List all pods in the federation.
        
        Returns:
            JSON string with pod list
        """
        federation = self.get_federation()
        pods = []
        
        for pod in federation:
            pods.append({
                "name": pod.name,
                "worker_count": pod.num_workers(),
                "inventory": pod.get_inventory_summary()
            })

        return json.dumps({
            "current_pod": self.get_pod().name if self.get_pod() else None,
            "pods": pods
        }, indent=2)
    
    # ===== Innovation Tools =====
    
    def invent_resource(self, name: str, base_value: float = 1.0, 
                        is_tool: bool = False, properties: Optional[Dict[str, float]] = None) -> str:
        """
        Invent a new resource type.
        
        Args:
            name: Name of the new resource
            base_value: Base market value
            is_tool: Whether this resource is a tool
            properties: Additional properties (quality, rarity, etc.)
        
        Returns:
            Status message
        """
        federation = self.get_federation()
        
        if federation.resource_registry.is_known(name):
            return json.dumps({
                "success": False,
                "error": f"Resource '{name}' already exists"
            }, indent=2)
        
        resource = federation.resource_registry.invent(
            name=name,
            inventor_id=self.worker.name,
            step=federation.steps,
            base_value=base_value,
            is_tool=is_tool,
            properties=properties or {}
        )
        
        return json.dumps({
            "success": True,
            "message": f"Invented new resource: {name}",
            "resource": {
                "name": resource.name,
                "base_value": resource.base_value,
                "is_tool": resource.is_tool
            }
        }, indent=2)
    
    def invent_recipe(self, name: str, steps: List[Dict], 
                      description: str = "", category: str = "general") -> str:
        """
        Invent a new production recipe.
        
        Args:
            name: Name of the new recipe
            steps: List of step definitions
            description: Recipe description
            category: Recipe category
        
        Returns:
            Status message
        """
        from ..economy import ProductionStep, StepType
        
        federation = self.get_federation()
        
        if federation.recipe_registry.get(name):
            return json.dumps({
                "success": False,
                "error": f"Recipe '{name}' already exists"
            }, indent=2)
        
        # Convert step dicts to ProductionStep objects
        recipe_steps = []
        for step_data in steps:
            step = ProductionStep(
                name=step_data['name'],
                step_type=StepType(step_data.get('step_type', 'processing')),
                duration=step_data['duration'],
                inputs=step_data.get('inputs', {}),
                outputs=step_data.get('outputs', {}),
                required_tool=step_data.get('required_tool'),
                required_skill=step_data.get('required_skill'),
                required_skill_level=step_data.get('required_skill_level', 1.0),
                requires_approval=step_data.get('requires_approval', False)
            )
            recipe_steps.append(step)
        
        recipe = federation.register_new_recipe(
            name=name,
            steps=recipe_steps,
            inventor_id=self.worker.unique_id,
            description=description,
            category=category
        )
        
        return json.dumps({
            "success": True,
            "message": f"Invented new recipe: {name}",
            "recipe": {
                "name": recipe.name,
                "category": recipe.category,
                "steps": len(recipe.steps),
                "duration": recipe.total_duration
            }
        }, indent=2)
    
    # ===== Status Tools =====
    
    def get_my_skills(self) -> str:
        """
        Get your current skill levels.
        
        Returns:
            JSON string with skill levels
        """
        return json.dumps({
            "skills": self.worker.skills
        }, indent=2)
    
    def get_federation_summary(self) -> str:
        """
        Get economic summary of the entire federation.
        
        Returns:
            JSON string with federation statistics
        """
        federation = self.get_federation()
        summary = federation.get_economic_summary()
        return json.dumps(summary, indent=2)

    # ===== Market Tools =====
    
    def get_market_price(self, resource_name: str) -> str:
        """
        Get current market price for a resource.
        
        Args:
            resource_name: Name of the resource
        
        Returns:
            JSON string with price information
        """
        federation = self.get_federation()
        market = federation.market
        
        if resource_name not in market.prices:
            return json.dumps({"error": f"Resource '{resource_name}' not registered in market"})
        
        price_info = market.prices[resource_name]
        return json.dumps({
            "resource": resource_name,
            "current_price": price_info.current_price,
            "base_price": price_info.base_price,
            "change_percent": ((price_info.current_price - price_info.base_price) / price_info.base_price * 100) if price_info.base_price > 0 else 0,
            "trend": price_info.trend
        }, indent=2)
    
    def buy_from_market(self, resource_name: str, quantity: float, max_price: float) -> str:
        """
        Place a buy order on the market.
        
        Args:
            resource_name: Resource to buy
            quantity: Amount to buy
            max_price: Maximum price per unit willing to pay
        
        Returns:
            Order status
        """
        federation = self.get_federation()
        market = federation.market
        
        # Check if resource is registered
        if resource_name not in market.prices:
            return json.dumps({"error": f"Resource '{resource_name}' not available on market"})
        
        # Check if worker has enough currency
        current_price = market.get_current_price(resource_name)
        estimated_cost = quantity * max_price
        
        if self.worker.currency < estimated_cost:
            return json.dumps({
                "success": False,
                "error": f"Insufficient funds. Need {estimated_cost:.2f}, have {self.worker.currency:.2f}"
            }, indent=2)
        
        # Place order
        current_step = self.worker._get_current_step()
        order = market.place_order(
            worker_id=self.worker.unique_id,
            pod_id=self.worker.pod.name if self.worker.pod else "unknown",
            resource_name=resource_name,
            quantity=quantity,
            price_limit=max_price,
            order_type="buy",
            timestamp=current_step
        )
        
        return json.dumps({
            "success": True,
            "order_id": order.order_id,
            "message": f"Buy order placed: {quantity} {resource_name} at max {max_price:.2f} each",
            "estimated_cost": estimated_cost
        }, indent=2)
    
    def sell_to_market(self, resource_name: str, quantity: float, min_price: float) -> str:
        """
        Place a sell order on the market.
        
        Args:
            resource_name: Resource to sell
            quantity: Amount to sell
            min_price: Minimum price per unit willing to accept
        
        Returns:
            Order status
        """
        federation = self.get_federation()
        market = federation.market
        pod = self.worker.pod
        
        # Check if resource is registered
        if resource_name not in market.prices:
            return json.dumps({"error": f"Resource '{resource_name}' not available on market"})
        
        # Check if pod has enough inventory
        if not pod:
            return json.dumps({"error": "Worker not assigned to a pod"}, indent=2)
        
        current_inventory = pod.inventory.get_quantity(resource_name)
        if current_inventory < quantity:
            return json.dumps({
                "success": False,
                "error": f"Insufficient inventory. Have {current_inventory}, need {quantity}"
            }, indent=2)
        
        # Place order
        current_step = self.worker._get_current_step()
        order = market.place_order(
            worker_id=self.worker.unique_id,
            pod_id=pod.unique_id,
            resource_name=resource_name,
            quantity=quantity,
            price_limit=min_price,
            order_type="sell",
            timestamp=current_step
        )
        
        return json.dumps({
            "success": True,
            "order_id": order.order_id,
            "message": f"Sell order placed: {quantity} {resource_name} at min {min_price:.2f} each",
            "estimated_revenue": quantity * min_price
        }, indent=2)
    
    def get_market_summary(self) -> str:
        """
        Get summary of current market conditions.
        
        Returns:
            JSON string with market summary
        """
        federation = self.get_federation()
        market = federation.market
        summary = market.get_market_summary()
        return json.dumps(summary, indent=2)
    
    def get_my_orders(self) -> str:
        """
        Get all active market orders for this worker.
        
        Returns:
            JSON string with order list
        """
        federation = self.get_federation()
        market = federation.market
        orders = market.get_worker_orders(self.worker.unique_id)
        
        order_list = []
        for order in orders:
            order_list.append({
                "order_id": order.order_id,
                "type": order.order_type,
                "resource": order.resource_name,
                "quantity": order.quantity - order.filled_quantity,
                "price_limit": order.price_limit,
                "filled": order.filled_quantity,
                "avg_price": order.average_price
            })
        
        return json.dumps({
            "active_orders": order_list,
            "total_active": len(order_list)
        }, indent=2)
    
    def cancel_order(self, order_id: str) -> str:
        """
        Cancel an active market order.
        
        Args:
            order_id: ID of the order to cancel
        
        Returns:
            Status message
        """
        federation = self.get_federation()
        market = federation.market
        
        if market.cancel_order(order_id):
            return json.dumps({
                "success": True,
                "message": f"Order {order_id} cancelled"
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "error": f"Order {order_id} not found or already filled"
            }, indent=2)
    
    def get_balance(self) -> str:
        """
        Get worker's current currency balance.
        
        Returns:
            JSON string with balance
        """
        return json.dumps({
            "worker_id": self.worker.name,
            "currency": self.worker.currency
        }, indent=2)


# Tool definitions for LLM function calling registration
def get_economic_tool_definitions() -> List[Dict]:
    """Get the tool definitions for OpenAI/LLM function calling."""
    
    return [
        {
            "type": "function",
            "function": {
                "name": "inspect_inventory",
                "description": "Check the current inventory of your pod",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_worker_status",
                "description": "Check your own status as a worker",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_known_recipes",
                "description": "List all known production recipes",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_recipe_details",
                "description": "Get detailed information about a specific recipe",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipe_name": {"type": "string", "description": "Name of the recipe"}
                    },
                    "required": ["recipe_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "start_production",
                "description": "Start producing a recipe in your pod",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipe_name": {"type": "string", "description": "Name of the recipe"},
                        "batch_size": {"type": "integer", "description": "Number of batches", "default": 1}
                    },
                    "required": ["recipe_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_my_tools",
                "description": "List all tools in your personal inventory",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "equip_tool",
                "description": "Equip a tool from your inventory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string", "description": "Name of the tool to equip"}
                    },
                    "required": ["tool_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "transfer_to_pod",
                "description": "Transfer resources from your pod to another pod",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "resource_name": {"type": "string", "description": "Resource to transfer"},
                        "quantity": {"type": "number", "description": "Amount to transfer"},
                        "target_pod_id": {"type": "string", "description": "Target pod ID"}
                    },
                    "required": ["resource_name", "quantity", "target_pod_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_pods",
                "description": "List all pods in the federation",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "invent_resource",
                "description": "Invent a new resource type",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the new resource"},
                        "base_value": {"type": "number", "description": "Base market value", "default": 1.0},
                        "is_tool": {"type": "boolean", "description": "Is this a tool?", "default": False},
                        "properties": {"type": "object", "description": "Additional properties"}
                    },
                    "required": ["name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_my_skills",
                "description": "Get your current skill levels",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "check_production_queue",
                "description": "Check the current production queue in your pod",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_market_price",
                "description": "Get current market price for a resource",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "resource_name": {"type": "string", "description": "Name of the resource"}
                    },
                    "required": ["resource_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "buy_from_market",
                "description": "Place a buy order on the market",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "resource_name": {"type": "string", "description": "Resource to buy"},
                        "quantity": {"type": "number", "description": "Amount to buy"},
                        "max_price": {"type": "number", "description": "Maximum price per unit"}
                    },
                    "required": ["resource_name", "quantity", "max_price"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "sell_to_market",
                "description": "Place a sell order on the market",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "resource_name": {"type": "string", "description": "Resource to sell"},
                        "quantity": {"type": "number", "description": "Amount to sell"},
                        "min_price": {"type": "number", "description": "Minimum price per unit"}
                    },
                    "required": ["resource_name", "quantity", "min_price"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_market_summary",
                "description": "Get summary of current market conditions",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_my_orders",
                "description": "Get all active market orders for this worker",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_order",
                "description": "Cancel an active market order",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "ID of the order to cancel"}
                    },
                    "required": ["order_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_balance",
                "description": "Get worker's current currency balance",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]