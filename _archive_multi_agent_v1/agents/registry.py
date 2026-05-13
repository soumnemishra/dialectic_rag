import logging
import threading
from typing import Optional, TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from src.agents.clinical_intent import ClinicalIntentModule
    from src.agents.safety_critic import ClinicalSafetyCriticAgent
    from src.agents.rag import RagAgent
    from src.agents.evidence_polarity_agent import EvidencePolarityModule
    from src.agents.evidence_governance import EvidenceGovernanceModule
    from src.tools.retriever import RetrieverTool

logger = logging.getLogger(__name__)

from src.config import settings

class AgentRegistry:
    """
    Singleton registry for all module and tool instances.
    """

    _instance: Optional["AgentRegistry"] = None
    _lock = threading.Lock()

    def __init__(self):
        if AgentRegistry._instance is not None:
            raise RuntimeError("AgentRegistry is a singleton. Use get_instance().")

        # Tools
        self._retriever: Optional["RetrieverTool"] = None
        self._dialectical_retriever = None

        # Modules
        self._clinical_intent:   Optional["ClinicalIntentModule"]       = None
        self._safety_critic:     Optional["ClinicalSafetyCriticAgent"] = None
        self._rag:               Optional["RagAgent"]                  = None
        self._evidence_polarity: Optional["EvidencePolarityModule"]     = None
        self._evidence_governance: Optional["EvidenceGovernanceModule"] = None
        self._nli_agent:         Optional["NliAgent"]                 = None

        self._initialized = False

    @classmethod
    def get_instance(cls) -> "AgentRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def initialize(self) -> None:
        if self._initialized:
            return

        logger.info("Initialising AgentRegistry (Epistemic Reasoning Pipeline)...")

        from src.tools.retriever import RetrieverTool, CrossEncoderReranker
        from src.tools.dense_retriever import DenseRetriever
        from src.core.registry import ModelRegistry

        embedding_device = ModelRegistry.get_embedding_device()
        dense_model = ModelRegistry.get_sentence_transformer(DenseRetriever.MEDICAL_MODEL, device=embedding_device)
        reranker_model = ModelRegistry.get_cross_encoder(CrossEncoderReranker.DEFAULT_MODEL, device=embedding_device)
        dedup_model = ModelRegistry.get_sentence_transformer("all-MiniLM-L6-v2", device=embedding_device)

        self._retriever = RetrieverTool(
            dense_retriever=DenseRetriever(use_medical=True, model=dense_model),
            reranker=CrossEncoderReranker(model=reranker_model),
        )

        if settings.EPISTEMIC_MODE and settings.ENABLE_DIALECTICAL_RETRIEVAL:
            try:
                from src.tools.dialectical_retriever import DialecticalRetriever
                self._dialectical_retriever = DialecticalRetriever(base_retriever=self._retriever, boost_factor=2.0)
            except Exception as e:
                logger.warning("Failed to create DialecticalRetriever: %s", e)

        from src.agents.clinical_intent       import ClinicalIntentModule
        from src.agents.safety_critic         import ClinicalSafetyCriticAgent
        from src.agents.rag                   import RagAgent
        from src.agents.evidence_polarity_agent  import EvidencePolarityModule
        from src.agents.evidence_governance   import EvidenceGovernanceModule

        self._clinical_intent  = ClinicalIntentModule()
        self._safety_critic    = ClinicalSafetyCriticAgent()
        self._rag              = RagAgent(retriever_tool=self._retriever)
        self._evidence_polarity = EvidencePolarityModule()
        self._evidence_governance = EvidenceGovernanceModule()

        try:
            from src.agents.nli_agent import NliAgent
            self._nli_agent = NliAgent(model_name=getattr(settings, "NLI_MODEL_NAME", None))
        except Exception:
            logger.warning("Failed to eagerly initialise NLI agent.")

        self._initialized = True

    def _lazy_init(self, attr: str, factory) -> object:
        val = getattr(self, attr)
        if val is None:
            if not self._initialized:
                self.initialize()
            val = getattr(self, attr)
            if val is None:
                val = factory()
                setattr(self, attr, val)
        return val

    @property
    def retriever(self) -> "RetrieverTool":
        from src.tools.retriever import RetrieverTool
        return self._lazy_init("_retriever", RetrieverTool)

    @property
    def dialectical_retriever(self):
        return getattr(self, "_dialectical_retriever", None)

    @property
    def clinical_intent(self) -> "ClinicalIntentModule":
        from src.agents.clinical_intent import ClinicalIntentModule
        return self._lazy_init("_clinical_intent", ClinicalIntentModule)

    @property
    def safety_critic(self) -> "ClinicalSafetyCriticAgent":
        from src.agents.safety_critic import ClinicalSafetyCriticAgent
        return self._lazy_init("_safety_critic", ClinicalSafetyCriticAgent)

    @property
    def rag(self) -> "RagAgent":
        from src.agents.rag import RagAgent
        return self._lazy_init("_rag", lambda: RagAgent(retriever_tool=self.retriever))

    @property
    def evidence_polarity(self) -> "EvidencePolarityModule":
        from src.agents.evidence_polarity_agent import EvidencePolarityModule
        return self._lazy_init("_evidence_polarity", EvidencePolarityModule)

    @property
    def evidence_governance(self) -> "EvidenceGovernanceModule":
        from src.agents.evidence_governance import EvidenceGovernanceModule
        return self._lazy_init("_evidence_governance", EvidenceGovernanceModule)

    @property
    def nli_agent(self):
        from src.agents.nli_agent import NliAgent
        return self._lazy_init("_nli_agent", NliAgent)