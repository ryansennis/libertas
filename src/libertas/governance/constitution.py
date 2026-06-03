"""Constitution system for defining organizational rules and permissions."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum, auto
import json


class ConstitutionLevel(Enum):
    """Level at which a constitution applies."""
    FEDERATION = auto()
    POD = auto()
    CUSTOM = auto()


@dataclass
class Article:
    """A single article in a constitution.

    Articles define specific rules, permissions, or requirements
    in natural language that can be read and interpreted by LLM agents.
    """

    article_id: str
    """Unique identifier (e.g., 'A1', 'A2')"""

    text: str
    """Natural language text of the article"""

    permissions: List[str] = field(default_factory=list)
    """List of permissions granted by this article (e.g., 'vote', 'propose')"""

    requirements: Dict[str, Any] = field(default_factory=dict)
    """Structured requirements (e.g., {'approval_threshold': 0.6, 'capital': 1000})"""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize article to dictionary."""
        return {
            "article_id": self.article_id,
            "text": self.text,
            "permissions": self.permissions.copy(),
            "requirements": self.requirements.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Article':
        """Create article from dictionary."""
        return cls(
            article_id=data["article_id"],
            text=data["text"],
            permissions=data.get("permissions", []),
            requirements=data.get("requirements", {}),
        )


@dataclass
class AmendmentProposal:
    """A proposed amendment to a constitution."""

    proposal_id: str
    """Unique identifier"""

    text: str
    """Text of the proposed amendment"""

    proposer: str
    """Worker ID of proposer"""

    required_support: float
    """Threshold for proposal to advance (0-1)"""

    supporters: List[str] = field(default_factory=list)
    """List of worker IDs supporting the proposal"""

    status: str = "proposed"
    """Status: proposed, voting, ratified, rejected"""

    created_step: int = 0
    """Simulation step when created"""

    def add_support(self, worker_id: str) -> bool:
        """Add support from a worker."""
        if worker_id not in self.supporters:
            self.supporters.append(worker_id)
            return True
        return False

    def has_sufficient_support(self, total_eligible: int) -> bool:
        """Check if proposal has enough support to advance."""
        if total_eligible == 0:
            return False
        support_ratio = len(self.supporters) / total_eligible
        return support_ratio >= self.required_support


@dataclass
class Constitution:
    """A living constitution that defines organizational rules and can be amended.

    Constitutions are written in natural language so LLM agents can read and
    interpret them to understand their rights, duties, and the rules governing
    their organization.
    """

    level: ConstitutionLevel
    """Level at which this constitution applies"""

    title: str
    """Title of the constitution"""

    preamble: str
    """Introductory text explaining purpose and values"""

    articles: List[Article] = field(default_factory=list)
    """List of articles defining specific rules"""

    amendment_process: Dict[str, Any] = field(default_factory=dict)
    """How amendments can be proposed and ratified"""

    version: int = 1
    """Current version number"""

    amendments: List[Dict[str, Any]] = field(default_factory=list)
    """History of amendments"""

    ratified_date: int = 0
    """Simulation step when ratified"""

    def get_full_text(self) -> str:
        """Return full constitutional text for LLM agents to read.

        This is the primary interface for agents - they read this text
        to understand what they can and cannot do.
        """
        text = f"{self.title}\n\n{self.preamble}\n\n"

        for article in self.articles:
            text += f"Article {article.article_id}: {article.text}\n\n"

        if self.amendment_process:
            text += "Amendment Process:\n"
            for key, value in self.amendment_process.items():
                formatted_key = key.replace('_', ' ').title()
                text += f"- {formatted_key}: {value}\n"

        return text.strip()

    def check_permission(self, agent: Any, action: str) -> Tuple[bool, str]:
        """Check if an action is permitted under this constitution.

        Args:
            agent: The agent attempting the action
            action: The action being attempted (e.g., 'vote', 'propose')

        Returns:
            Tuple of (allowed, reason)
        """
        # Check articles for explicit permissions
        for article in self.articles:
            if action in article.permissions:
                # Check if agent meets requirements
                if article.requirements:
                    # Example requirement checks
                    if "min_currency" in article.requirements:
                        if hasattr(agent, 'currency'):
                            if agent.currency < article.requirements["min_currency"]:
                                return False, f"Insufficient currency (need {article.requirements['min_currency']})"

                    if "required_skill" in article.requirements:
                        skill = article.requirements["required_skill"]
                        if hasattr(agent, 'get_skill_level'):
                            if agent.get_skill_level(skill) < article.requirements.get("min_skill_level", 1.0):
                                return False, f"Insufficient {skill} skill"

                # All requirements met
                return True, f"Permitted by Article {article.article_id}"

        # No explicit permission found
        return False, f"Action '{action}' not explicitly permitted in constitution"

    def propose_amendment(
        self,
        proposer: str,
        amendment_text: str,
        proposal_id: str,
        current_step: int = 0
    ) -> AmendmentProposal:
        """Create an amendment proposal.

        Args:
            proposer: Worker ID of proposer
            amendment_text: Text of proposed amendment
            proposal_id: Unique identifier for this proposal
            current_step: Current simulation step

        Returns:
            AmendmentProposal object
        """
        required_support = self.amendment_process.get("proposal_threshold", 0.1)

        proposal = AmendmentProposal(
            proposal_id=proposal_id,
            text=amendment_text,
            proposer=proposer,
            required_support=required_support,
            created_step=current_step,
        )

        # Proposer automatically supports
        proposal.add_support(proposer)

        return proposal

    def ratify_amendment(
        self,
        amendment_text: str,
        vote_count: Dict[str, int],
        current_step: int = 0
    ) -> bool:
        """Ratify an amendment if it passes the required threshold.

        Args:
            amendment_text: Text of the amendment
            vote_count: Dictionary with 'for' and 'against' counts
            current_step: Current simulation step

        Returns:
            True if amendment was ratified
        """
        total_votes = vote_count.get('for', 0) + vote_count.get('against', 0)
        if total_votes == 0:
            return False

        support_ratio = vote_count.get('for', 0) / total_votes
        required_threshold = self.amendment_process.get("ratification_threshold", 0.67)

        if support_ratio >= required_threshold:
            # Add to amendments history
            self.amendments.append({
                "text": amendment_text,
                "ratified_step": current_step,
                "version": self.version + 1,
                "vote_count": vote_count.copy(),
            })

            self.version += 1
            return True

        return False

    def get_article(self, article_id: str) -> Optional[Article]:
        """Get an article by its ID."""
        for article in self.articles:
            if article.article_id == article_id:
                return article
        return None

    def add_article(self, article: Article) -> None:
        """Add a new article to the constitution."""
        self.articles.append(article)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize constitution to dictionary."""
        return {
            "level": self.level.name,
            "title": self.title,
            "preamble": self.preamble,
            "articles": [article.to_dict() for article in self.articles],
            "amendment_process": self.amendment_process.copy(),
            "version": self.version,
            "amendments": [a.copy() for a in self.amendments],
            "ratified_date": self.ratified_date,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Constitution':
        """Create constitution from dictionary."""
        return cls(
            level=ConstitutionLevel[data["level"]],
            title=data["title"],
            preamble=data["preamble"],
            articles=[Article.from_dict(a) for a in data.get("articles", [])],
            amendment_process=data.get("amendment_process", {}),
            version=data.get("version", 1),
            amendments=data.get("amendments", []),
            ratified_date=data.get("ratified_date", 0),
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'Constitution':
        """Create constitution from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def create_default_federation_constitution(cls) -> 'Constitution':
        """Create a default federation constitution."""
        return cls(
            level=ConstitutionLevel.FEDERATION,
            title="Federation of Worker Cooperatives - Constitution",
            preamble="We, the workers, establish this federation to promote economic democracy, "
                     "mutual aid, and collective prosperity. This constitution defines our "
                     "shared rights, responsibilities, and decision-making processes.",
            articles=[
                Article(
                    article_id="A1",
                    text="All workers have equal voting rights regardless of role, tenure, or skill level.",
                    permissions=["vote"],
                ),
                Article(
                    article_id="A2",
                    text="Any worker may propose motions for consideration by the federation.",
                    permissions=["propose_motion"],
                ),
                Article(
                    article_id="A3",
                    text="Workers may form new pods with approval of 60% of federation members "
                         "and contribution of 1000 currency units as initial capital.",
                    permissions=["create_pod"],
                    requirements={"approval_threshold": 0.6, "capital": 1000},
                ),
                Article(
                    article_id="A4",
                    text="Amendments may be proposed with support of at least 10% of workers.",
                    permissions=["propose_amendment"],
                ),
            ],
            amendment_process={
                "proposal_threshold": 0.1,
                "ratification_threshold": 0.67,
            },
        )

    @classmethod
    def create_default_pod_constitution(cls, pod_name: str = "Worker Pod") -> 'Constitution':
        """Create a default pod constitution (direct democracy model)."""
        return cls(
            level=ConstitutionLevel.POD,
            title=f"{pod_name} - Constitution",
            preamble=f"We, the members of {pod_name}, commit to operating as a direct democracy "
                     "where all members have equal say in decisions affecting our collective work.",
            articles=[
                Article(
                    article_id="A1",
                    text="All decisions require simple majority approval of active members.",
                    permissions=["vote"],
                    requirements={"approval_threshold": 0.5},
                ),
                Article(
                    article_id="A2",
                    text="Profits are shared equally among all active members.",
                    permissions=["receive_profit"],
                ),
                Article(
                    article_id="A3",
                    text="Any member may propose production plans or resource allocations.",
                    permissions=["propose_production", "propose_allocation"],
                ),
                Article(
                    article_id="A4",
                    text="Members may access shared tools and inventory for authorized work.",
                    permissions=["access_inventory", "use_tools"],
                ),
            ],
            amendment_process={
                "proposal_threshold": 0.2,
                "ratification_threshold": 0.6,
            },
        )
