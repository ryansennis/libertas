"""Unit tests for constitution system."""

import pytest
from libertas.governance import (
    Constitution,
    Article,
    ConstitutionLevel,
    AmendmentProposal,
)


@pytest.mark.unit
class TestArticle:
    """Test Article class."""

    def test_article_creation(self):
        """Test creating an article."""
        article = Article(
            article_id="A1",
            text="All workers have equal voting rights.",
            permissions=["vote", "propose"],
            requirements={"min_currency": 100}
        )

        assert article.article_id == "A1"
        assert "voting rights" in article.text
        assert "vote" in article.permissions
        assert article.requirements["min_currency"] == 100

    def test_article_to_dict(self):
        """Test serializing article to dict."""
        article = Article(
            article_id="A2",
            text="Workers may form pods.",
            permissions=["create_pod"],
            requirements={"capital": 1000}
        )

        data = article.to_dict()

        assert data["article_id"] == "A2"
        assert data["text"] == "Workers may form pods."
        assert data["permissions"] == ["create_pod"]
        assert data["requirements"] == {"capital": 1000}

    def test_article_from_dict(self):
        """Test creating article from dict."""
        data = {
            "article_id": "A3",
            "text": "Test article",
            "permissions": ["test"],
            "requirements": {"test_req": 50}
        }

        article = Article.from_dict(data)

        assert article.article_id == "A3"
        assert article.text == "Test article"
        assert article.permissions == ["test"]
        assert article.requirements == {"test_req": 50}

    def test_article_from_dict_missing_optional(self):
        """Test creating article from dict without optional fields."""
        data = {
            "article_id": "A4",
            "text": "Simple article"
        }

        article = Article.from_dict(data)

        assert article.article_id == "A4"
        assert article.permissions == []
        assert article.requirements == {}


@pytest.mark.unit
class TestAmendmentProposal:
    """Test AmendmentProposal class."""

    def test_proposal_creation(self):
        """Test creating an amendment proposal."""
        proposal = AmendmentProposal(
            proposal_id="P1",
            text="Increase voting threshold to 75%",
            proposer="worker_001",
            required_support=0.1
        )

        assert proposal.proposal_id == "P1"
        assert proposal.proposer == "worker_001"
        assert proposal.required_support == 0.1
        assert proposal.status == "proposed"
        assert len(proposal.supporters) == 0

    def test_add_support(self):
        """Test adding support to proposal."""
        proposal = AmendmentProposal(
            proposal_id="P2",
            text="Test amendment",
            proposer="worker_001",
            required_support=0.2
        )

        # Add first supporter
        result = proposal.add_support("worker_002")
        assert result is True
        assert "worker_002" in proposal.supporters

        # Try to add same supporter again
        result = proposal.add_support("worker_002")
        assert result is False
        assert len(proposal.supporters) == 1

    def test_has_sufficient_support(self):
        """Test checking if proposal has enough support."""
        proposal = AmendmentProposal(
            proposal_id="P3",
            text="Test amendment",
            proposer="worker_001",
            required_support=0.3
        )

        # Add 2 supporters out of 10 eligible (20% < 30%)
        proposal.add_support("worker_002")
        proposal.add_support("worker_003")
        assert proposal.has_sufficient_support(10) is False

        # Add 1 more (30% = 30%)
        proposal.add_support("worker_004")
        assert proposal.has_sufficient_support(10) is True

        # Add 1 more (40% > 30%)
        proposal.add_support("worker_005")
        assert proposal.has_sufficient_support(10) is True

    def test_has_sufficient_support_zero_eligible(self):
        """Test support check with zero eligible voters."""
        proposal = AmendmentProposal(
            proposal_id="P4",
            text="Test",
            proposer="worker_001",
            required_support=0.1
        )

        assert proposal.has_sufficient_support(0) is False


@pytest.mark.unit
class TestConstitution:
    """Test Constitution class."""

    def test_constitution_creation(self):
        """Test creating a basic constitution."""
        constitution = Constitution(
            level=ConstitutionLevel.FEDERATION,
            title="Test Constitution",
            preamble="We the workers...",
            articles=[
                Article("A1", "All workers can vote", ["vote"], {})
            ]
        )

        assert constitution.level == ConstitutionLevel.FEDERATION
        assert constitution.title == "Test Constitution"
        assert constitution.preamble == "We the workers..."
        assert len(constitution.articles) == 1
        assert constitution.version == 1

    def test_get_full_text(self):
        """Test getting formatted constitution text."""
        constitution = Constitution(
            level=ConstitutionLevel.POD,
            title="Pod Constitution",
            preamble="This pod operates democratically.",
            articles=[
                Article("A1", "All members vote equally.", ["vote"], {}),
                Article("A2", "Profits shared equally.", ["receive_profit"], {})
            ],
            amendment_process={
                "proposal_threshold": 0.2,
                "ratification_threshold": 0.6
            }
        )

        text = constitution.get_full_text()

        assert "Pod Constitution" in text
        assert "This pod operates democratically." in text
        assert "Article A1: All members vote equally." in text
        assert "Article A2: Profits shared equally." in text
        assert "Proposal Threshold: 0.2" in text
        assert "Ratification Threshold: 0.6" in text

    def test_check_permission_granted(self):
        """Test permission check when action is permitted."""
        constitution = Constitution(
            level=ConstitutionLevel.POD,
            title="Test",
            preamble="Test",
            articles=[
                Article("A1", "All can vote", ["vote"], {})
            ]
        )

        # Mock agent
        class MockAgent:
            pass

        agent = MockAgent()
        allowed, reason = constitution.check_permission(agent, "vote")

        assert allowed is True
        assert "Article A1" in reason

    def test_check_permission_denied_not_found(self):
        """Test permission check when action not in constitution."""
        constitution = Constitution(
            level=ConstitutionLevel.POD,
            title="Test",
            preamble="Test",
            articles=[
                Article("A1", "All can vote", ["vote"], {})
            ]
        )

        class MockAgent:
            pass

        agent = MockAgent()
        allowed, reason = constitution.check_permission(agent, "manage")

        assert allowed is False
        assert "not explicitly permitted" in reason

    def test_check_permission_denied_insufficient_currency(self):
        """Test permission denied due to currency requirement."""
        constitution = Constitution(
            level=ConstitutionLevel.FEDERATION,
            title="Test",
            preamble="Test",
            articles=[
                Article(
                    "A1",
                    "Workers with 1000+ currency can create pods",
                    ["create_pod"],
                    {"min_currency": 1000}
                )
            ]
        )

        class MockAgent:
            currency = 500

        agent = MockAgent()
        allowed, reason = constitution.check_permission(agent, "create_pod")

        assert allowed is False
        assert "Insufficient currency" in reason
        assert "1000" in reason

    def test_check_permission_granted_with_currency(self):
        """Test permission granted when currency requirement met."""
        constitution = Constitution(
            level=ConstitutionLevel.FEDERATION,
            title="Test",
            preamble="Test",
            articles=[
                Article(
                    "A1",
                    "Workers with 1000+ currency can create pods",
                    ["create_pod"],
                    {"min_currency": 1000}
                )
            ]
        )

        class MockAgent:
            currency = 1500

        agent = MockAgent()
        allowed, reason = constitution.check_permission(agent, "create_pod")

        assert allowed is True
        assert "Article A1" in reason

    def test_check_permission_denied_insufficient_skill(self):
        """Test permission denied due to skill requirement."""
        constitution = Constitution(
            level=ConstitutionLevel.POD,
            title="Test",
            preamble="Test",
            articles=[
                Article(
                    "A1",
                    "Skilled workers can manage production",
                    ["manage_production"],
                    {"required_skill": "manufacturing", "min_skill_level": 5.0}
                )
            ]
        )

        class MockAgent:
            def get_skill_level(self, skill):
                return 3.0

        agent = MockAgent()
        allowed, reason = constitution.check_permission(agent, "manage_production")

        assert allowed is False
        assert "Insufficient manufacturing skill" in reason

    def test_check_permission_granted_with_skill(self):
        """Test permission granted when skill requirement met."""
        constitution = Constitution(
            level=ConstitutionLevel.POD,
            title="Test",
            preamble="Test",
            articles=[
                Article(
                    "A1",
                    "Skilled workers can manage",
                    ["manage"],
                    {"required_skill": "leadership", "min_skill_level": 4.0}
                )
            ]
        )

        class MockAgent:
            def get_skill_level(self, skill):
                return 5.0

        agent = MockAgent()
        allowed, reason = constitution.check_permission(agent, "manage")

        assert allowed is True

    def test_propose_amendment(self):
        """Test proposing a constitutional amendment."""
        constitution = Constitution(
            level=ConstitutionLevel.FEDERATION,
            title="Test",
            preamble="Test",
            articles=[],
            amendment_process={"proposal_threshold": 0.15}
        )

        proposal = constitution.propose_amendment(
            proposer="worker_001",
            amendment_text="Increase profit sharing to 80%",
            proposal_id="P1",
            current_step=100
        )

        assert proposal.proposal_id == "P1"
        assert proposal.proposer == "worker_001"
        assert proposal.text == "Increase profit sharing to 80%"
        assert proposal.required_support == 0.15
        assert proposal.created_step == 100
        assert "worker_001" in proposal.supporters  # Proposer auto-supports

    def test_ratify_amendment_passes(self):
        """Test ratifying an amendment that passes."""
        constitution = Constitution(
            level=ConstitutionLevel.POD,
            title="Test",
            preamble="Test",
            articles=[],
            amendment_process={"ratification_threshold": 0.67}
        )

        # 70% vote for (passes 67% threshold)
        vote_count = {"for": 70, "against": 30}
        result = constitution.ratify_amendment(
            amendment_text="New profit sharing rule",
            vote_count=vote_count,
            current_step=200
        )

        assert result is True
        assert constitution.version == 2
        assert len(constitution.amendments) == 1
        assert constitution.amendments[0]["text"] == "New profit sharing rule"
        assert constitution.amendments[0]["ratified_step"] == 200
        assert constitution.amendments[0]["version"] == 2

    def test_ratify_amendment_fails(self):
        """Test ratifying an amendment that fails."""
        constitution = Constitution(
            level=ConstitutionLevel.FEDERATION,
            title="Test",
            preamble="Test",
            articles=[],
            amendment_process={"ratification_threshold": 0.67}
        )

        # 60% vote for (fails 67% threshold)
        vote_count = {"for": 60, "against": 40}
        result = constitution.ratify_amendment(
            amendment_text="Failed amendment",
            vote_count=vote_count,
            current_step=300
        )

        assert result is False
        assert constitution.version == 1
        assert len(constitution.amendments) == 0

    def test_ratify_amendment_no_votes(self):
        """Test ratifying with no votes cast."""
        constitution = Constitution(
            level=ConstitutionLevel.POD,
            title="Test",
            preamble="Test",
            articles=[]
        )

        vote_count = {"for": 0, "against": 0}
        result = constitution.ratify_amendment(
            amendment_text="No votes",
            vote_count=vote_count,
            current_step=100
        )

        assert result is False

    def test_get_article(self):
        """Test retrieving article by ID."""
        article1 = Article("A1", "First article", [], {})
        article2 = Article("A2", "Second article", [], {})

        constitution = Constitution(
            level=ConstitutionLevel.POD,
            title="Test",
            preamble="Test",
            articles=[article1, article2]
        )

        found = constitution.get_article("A2")
        assert found is not None
        assert found.article_id == "A2"
        assert found.text == "Second article"

        not_found = constitution.get_article("A99")
        assert not_found is None

    def test_add_article(self):
        """Test adding article to constitution."""
        constitution = Constitution(
            level=ConstitutionLevel.POD,
            title="Test",
            preamble="Test",
            articles=[]
        )

        assert len(constitution.articles) == 0

        article = Article("A1", "New article", ["new_permission"], {})
        constitution.add_article(article)

        assert len(constitution.articles) == 1
        assert constitution.articles[0].article_id == "A1"

    def test_to_dict(self):
        """Test serializing constitution to dict."""
        constitution = Constitution(
            level=ConstitutionLevel.FEDERATION,
            title="Test Constitution",
            preamble="Preamble text",
            articles=[
                Article("A1", "Article 1 text", ["vote"], {"min_currency": 100})
            ],
            amendment_process={"proposal_threshold": 0.1},
            version=2,
            amendments=[{"text": "Past amendment", "ratified_step": 50}],
            ratified_date=10
        )

        data = constitution.to_dict()

        assert data["level"] == "FEDERATION"
        assert data["title"] == "Test Constitution"
        assert data["preamble"] == "Preamble text"
        assert len(data["articles"]) == 1
        assert data["articles"][0]["article_id"] == "A1"
        assert data["amendment_process"]["proposal_threshold"] == 0.1
        assert data["version"] == 2
        assert len(data["amendments"]) == 1
        assert data["ratified_date"] == 10

    def test_from_dict(self):
        """Test creating constitution from dict."""
        data = {
            "level": "POD",
            "title": "Test Pod",
            "preamble": "Test preamble",
            "articles": [
                {"article_id": "A1", "text": "Article 1", "permissions": [], "requirements": {}}
            ],
            "amendment_process": {"proposal_threshold": 0.2},
            "version": 3,
            "amendments": [{"text": "Amendment 1"}],
            "ratified_date": 20
        }

        constitution = Constitution.from_dict(data)

        assert constitution.level == ConstitutionLevel.POD
        assert constitution.title == "Test Pod"
        assert constitution.preamble == "Test preamble"
        assert len(constitution.articles) == 1
        assert constitution.articles[0].article_id == "A1"
        assert constitution.amendment_process["proposal_threshold"] == 0.2
        assert constitution.version == 3
        assert len(constitution.amendments) == 1
        assert constitution.ratified_date == 20

    def test_to_json_and_from_json(self):
        """Test JSON serialization round-trip."""
        original = Constitution(
            level=ConstitutionLevel.FEDERATION,
            title="Original",
            preamble="Test",
            articles=[Article("A1", "Test", ["vote"], {})],
            amendment_process={"proposal_threshold": 0.15}
        )

        json_str = original.to_json()
        restored = Constitution.from_json(json_str)

        assert restored.level == original.level
        assert restored.title == original.title
        assert restored.preamble == original.preamble
        assert len(restored.articles) == len(original.articles)
        assert restored.articles[0].article_id == original.articles[0].article_id

    def test_create_default_federation_constitution(self):
        """Test creating default federation constitution."""
        constitution = Constitution.create_default_federation_constitution()

        assert constitution.level == ConstitutionLevel.FEDERATION
        assert "Federation" in constitution.title
        assert len(constitution.articles) >= 4
        assert constitution.amendment_process["proposal_threshold"] == 0.1
        assert constitution.amendment_process["ratification_threshold"] == 0.67

        # Check for key articles
        article_ids = [a.article_id for a in constitution.articles]
        assert "A1" in article_ids
        assert "A2" in article_ids
        assert "A3" in article_ids
        assert "A4" in article_ids

        # Check permissions
        all_permissions = []
        for article in constitution.articles:
            all_permissions.extend(article.permissions)
        assert "vote" in all_permissions
        assert "propose_motion" in all_permissions
        assert "create_pod" in all_permissions

    def test_create_default_pod_constitution(self):
        """Test creating default pod constitution."""
        constitution = Constitution.create_default_pod_constitution("Test Pod")

        assert constitution.level == ConstitutionLevel.POD
        assert "Test Pod" in constitution.title
        assert len(constitution.articles) >= 4

        # Check for democracy-related articles
        full_text = constitution.get_full_text()
        assert "majority" in full_text.lower() or "democracy" in full_text.lower()

        # Check for key permissions
        all_permissions = []
        for article in constitution.articles:
            all_permissions.extend(article.permissions)
        assert "vote" in all_permissions
        assert "receive_profit" in all_permissions
