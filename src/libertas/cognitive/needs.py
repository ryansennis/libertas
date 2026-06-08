"""Worker needs system - physiological and lifestyle needs.

This module tracks worker needs that degrade over time and affect mood.
Workers make purchasing decisions via LLM based on their needs state.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class WorkerNeeds:
    """
    Track worker's physiological and lifestyle needs.
    These affect mood and drive purchasing decisions.

    Needs degrade over time and must be satisfied through purchases or rest.
    Low needs negatively affect mood, creating economic pressure.
    """

    # Physiological needs (0-1 scale, 0=critical, 1=fully satisfied)
    hunger: float = 0.8
    thirst: float = 0.8
    rest: float = 0.8
    recreation: float = 0.7  # Entertainment, social activities

    # Lifestyle/economic needs (abstracted - represents home comfort)
    housing_satisfaction: float = 0.7  # Home furnishings, living conditions

    # Degradation rates (how fast needs decline per step)
    hunger_decay: float = 0.05
    thirst_decay: float = 0.08
    rest_decay: float = 0.03
    recreation_decay: float = 0.02
    housing_decay: float = 0.01  # Slow - represents gradual wear

    # Purchase history (for learning and memory)
    recent_purchases: List[Dict[str, Any]] = field(default_factory=list)

    def degrade_needs(self) -> None:
        """Degrade needs over time (called each simulation step)."""
        self.hunger = max(0.0, self.hunger - self.hunger_decay)
        self.thirst = max(0.0, self.thirst - self.thirst_decay)
        self.rest = max(0.0, self.rest - self.rest_decay)
        self.recreation = max(0.0, self.recreation - self.recreation_decay)
        self.housing_satisfaction = max(0.0, self.housing_satisfaction - self.housing_decay)

    def get_critical_needs(self) -> List[str]:
        """Return list of needs below critical threshold (0.3)."""
        critical = []
        if self.hunger < 0.3:
            critical.append("hunger")
        if self.thirst < 0.3:
            critical.append("thirst")
        if self.rest < 0.3:
            critical.append("rest")
        if self.recreation < 0.2:  # Lower threshold - less critical
            critical.append("recreation")
        if self.housing_satisfaction < 0.2:
            critical.append("housing")
        return critical

    def get_needs_summary(self) -> str:
        """Natural language summary of needs for LLM context."""
        summary = []

        if self.hunger < 0.3:
            summary.append("very hungry")
        elif self.hunger < 0.6:
            summary.append("somewhat hungry")

        if self.thirst < 0.3:
            summary.append("very thirsty")
        elif self.thirst < 0.6:
            summary.append("somewhat thirsty")

        if self.rest < 0.3:
            summary.append("exhausted")
        elif self.rest < 0.6:
            summary.append("tired")

        if self.recreation < 0.3:
            summary.append("bored and restless")
        elif self.recreation < 0.5:
            summary.append("wanting entertainment")

        if self.housing_satisfaction < 0.3:
            summary.append("dissatisfied with living conditions")

        if not summary:
            return "All needs are satisfied"

        return "You are " + ", ".join(summary)

    def affect_mood(self, mood: 'MoodState') -> None:
        """Unmet needs negatively affect mood.

        This is called during worker.step() to update mood based on needs.
        """
        # Hunger affects happiness and motivation
        if self.hunger < 0.3:
            mood.happiness -= 0.05
            mood.motivation -= 0.03

        # Thirst affects stress
        if self.thirst < 0.3:
            mood.stress += 0.05

        # Rest affects stress and motivation
        if self.rest < 0.3:
            mood.stress += 0.05
            mood.motivation -= 0.05

        # Recreation affects happiness and solidarity (social need)
        if self.recreation < 0.3:
            mood.happiness -= 0.03
            mood.solidarity_with_group -= 0.02

        # Housing affects happiness and stress (comfort/security)
        if self.housing_satisfaction < 0.3:
            mood.happiness -= 0.02
            mood.stress += 0.02

    def consume_food(self, food_value: float = 0.4) -> None:
        """Eat food, satisfy hunger."""
        self.hunger = min(1.0, self.hunger + food_value)

    def consume_water(self, water_value: float = 0.5) -> None:
        """Drink water, satisfy thirst."""
        self.thirst = min(1.0, self.thirst + water_value)

    def sleep(self, sleep_value: float = 0.6) -> None:
        """Sleep, restore rest."""
        self.rest = min(1.0, self.rest + sleep_value)

    def enjoy_recreation(self, recreation_value: float = 0.3) -> None:
        """Enjoy entertainment/recreation."""
        self.recreation = min(1.0, self.recreation + recreation_value)

    def improve_housing(self, improvement_value: float = 0.2) -> None:
        """Purchase home furnishings/improvements (abstracted)."""
        self.housing_satisfaction = min(1.0, self.housing_satisfaction + improvement_value)

    def record_purchase(self, item_name: str, cost: float, need_satisfied: str) -> None:
        """Record a purchase for memory/learning."""
        self.recent_purchases.append({
            'item': item_name,
            'cost': cost,
            'need': need_satisfied
        })
        # Keep only last 10 purchases
        if len(self.recent_purchases) > 10:
            self.recent_purchases.pop(0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize needs state."""
        return {
            'hunger': self.hunger,
            'thirst': self.thirst,
            'rest': self.rest,
            'recreation': self.recreation,
            'housing_satisfaction': self.housing_satisfaction,
            'hunger_decay': self.hunger_decay,
            'thirst_decay': self.thirst_decay,
            'rest_decay': self.rest_decay,
            'recreation_decay': self.recreation_decay,
            'housing_decay': self.housing_decay,
            'recent_purchases': self.recent_purchases.copy()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkerNeeds':
        """Deserialize needs state."""
        return cls(
            hunger=data.get('hunger', 0.8),
            thirst=data.get('thirst', 0.8),
            rest=data.get('rest', 0.8),
            recreation=data.get('recreation', 0.7),
            housing_satisfaction=data.get('housing_satisfaction', 0.7),
            hunger_decay=data.get('hunger_decay', 0.05),
            thirst_decay=data.get('thirst_decay', 0.08),
            rest_decay=data.get('rest_decay', 0.03),
            recreation_decay=data.get('recreation_decay', 0.02),
            housing_decay=data.get('housing_decay', 0.01),
            recent_purchases=data.get('recent_purchases', []).copy()
        )
