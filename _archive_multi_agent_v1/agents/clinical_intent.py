###################################### tools that we will be using ############################
from typing import Dict, Any  #python way of saying this is dict 
from langchain_core.prompts import ChatPromptTemplate # a template for talking to ai 
from langchain_core.output_parsers import JsonOutputParser #makes ai to give clean  structure answer 
#the graph state is the shared note book 
from src.state.state import GraphState, ClinicalIntentFormat #this is the shared memory accross the agents  and each agent read from this file 
from src.prompts.templates import CLINICAL_INTENT_SYSTEM_PROMPT, CLINICAL_INTENT_HUMAN_PROMPT, with_json_system_suffix, format_chat_history #this contain actual instrcutions 
from src.core.registry import ModelRegistry, safe_ainvoke #the warehouse that holds the model 
from src.query_builder import parse_markdown_json
import logging #for keeping records of what happened 
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError  # this basically solves api timeout etc 

logger = logging.getLogger(__name__)



############# Think of this as the module's functional logic ############# 
class ClinicalIntentModule:
    """
    Module responsible for classifying medical intent and risk level of queries.
    This is the first gate in the Epistemic Reasoning Pipeline.
    """
    #this is setuping the agent desk 
    def __init__(self):
        # Use a fast model for intent classification to minimize latency
         #this is just like ticking the boxes on form 
         #get the llm brain ready selm.llm get ai brain ready 
         #model registry says go to the warehouse and get me the flash llm (small fast model)
         #temperature is set to 0.0 to make the model more deterministic and less creative
         #json_mode=true forces the brain to speak in structured format not in setence 
        self.llm = ModelRegistry.get_flash_llm(temperature=0.0, json_mode=True)#we are assigning here a small model 
        if self.llm is None:
            raise RuntimeError("ClinicalIntentAgent: Flash LLM failed to load. Check ModelRegistry.")
        #because intent classification is a simple task and does not require a large model
        #the temperature is set to 0.0 to make the model more deterministic and less creative
        #this json_mode=true forces the brain to speak in structured format not in setence 

        # it takes the  json output of the llm and compare it with clinical itent format 
        #so that it doesnot breaks the shared state doesnot breaks 
        self.parser = JsonOutputParser(pydantic_object=ClinicalIntentFormat)#this just instruct the llm to fll out the form 
        #it just fill and pass to the next agent so that it doesnot breaks the shared state doesnot breaks 

        #####  below think of like the instructions for the agent 
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", with_json_system_suffix(CLINICAL_INTENT_SYSTEM_PROMPT)),
            ("human", "{human_content}")
        ])

        # Build chain once to avoid rebuilding on every retry
        self.chain = self.prompt | self.llm | self.parser
        self.raw_chain = self.prompt | self.llm
        logger.info("ClinicalIntentModule chain initialized")

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.5))
    async def _classify_with_retry(self, human_content: str) -> dict:
        try:
            return await safe_ainvoke(self.chain, {"human_content": human_content})
        except RetryError:
            raw_result = await safe_ainvoke(self.raw_chain, {"human_content": human_content})
            raw_text = raw_result.content if hasattr(raw_result, "content") else str(raw_result)
            return parse_markdown_json(raw_text)
        #async create agent reusability si that agent is initialized once and not all the time this saves time and computation
    async def classify(self, state: GraphState) -> Dict[str, Any]:
        """
        Classify the intent of the question.
        Returns a dictionary with keys matching the GraphState fields to update.
        ainvoke is the "Start" button. It pushes the question into the first station.
        await is the "Waiting Room." Because the LLM "Brain" takes a few seconds to think, await tells the rest of your computer: 
        "Hey, go do other chores while we wait for this brain to finish. Don't just sit here frozen.
        """
        try:
            question = state["original_question"]
            chat_history_raw = state.get("chat_history", [])
            formatted_history = format_chat_history(chat_history_raw) if chat_history_raw else ""
            logger.info(f"Classifying intent for: {question}")

            try:
                human_content = CLINICAL_INTENT_HUMAN_PROMPT.format(
                    question=question,
                )
            except KeyError:
                human_content = CLINICAL_INTENT_HUMAN_PROMPT.format(
                    question=question,
                    chat_history="",
                )
            if formatted_history.strip():
                human_content += f"\n\nPrior Conversation History:\n{formatted_history}"

            try:
                result = await self._classify_with_retry(human_content)
            except RetryError:
                raw_result = await safe_ainvoke(self.raw_chain, {"human_content": human_content})
                raw_text = raw_result.content if hasattr(raw_result, "content") else str(raw_result)
                result = parse_markdown_json(raw_text)
            
            logger.info(f"Intent classification complete: {result}")
            
            # Return update for the state
            return {
                "intent": result.get("intent", "informational"),
                "risk_level": result.get("risk_level", "high"),
                "requires_disclaimer": result.get("requires_disclaimer", True),
                "needs_guidelines": result.get("needs_guidelines", True),
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", ""),
                "safety_flags": []
            }
            
        except Exception as e:
            logger.error(f"Failed after retries: {e}")
            # Fallback to safe defaults
            return {
                "intent": "informational", 
                "risk_level": "high",
                "requires_disclaimer": True,
                "needs_guidelines": True,
                "confidence": 0.0,
                "reasoning": "Fallback triggered due to error",
                "safety_flags": ["classification_error_safe_fallback"]
            }


from src.agents.registry import AgentRegistry
#The Integration (clinical_intent_node)
#This is a "wrapper" function. 
# #It’s what your StateGraph actually calls. 
# It pulls the module from a Registry (a warehouse for your modules) 
# so you don't keep recreating the same module over and over, saving memory.
async def clinical_intent_node(state: GraphState) -> Dict[str, Any]:
    try:
        module = AgentRegistry.get_instance().clinical_intent
        return await module.classify(state)
    except Exception as e:
        logger.critical(f"clinical_intent_node failed: {e}")
        return {
            "intent": "informational",
            "risk_level": "high",
            "requires_disclaimer": True,
            "needs_guidelines": True,
            "confidence": 0.0,
            "reasoning": "Fallback triggered due to error",
            "safety_flags": ["node_level_error"]
        }

# "Each module in my system works like a medical specialist.
# They all read from a shared state and only add their own findings, which makes the system safe, debuggable, and modular."