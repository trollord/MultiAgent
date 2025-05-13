import streamlit as st
import asyncio
import os

# --- ADK Imports ---
# Core components for agent, model interaction, session management, and execution
from google.adk.agents import Agent
from google.adk.tools import google_search 
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService, Session
from google.adk.runners import Runner
from google.genai import types as genai_types
# Tool and callback related imports
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools import agent_tool

# Standard Python libraries
import warnings
import logging

# --- Asyncio Configuration for Streamlit ---
import nest_asyncio
nest_asyncio.apply()

# --- Basic Configuration ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Medical Coding Assistant", layout="wide")
st.title("🩺 Medical Coding Assistant")
st.caption("Process patient discharge summaries to extract ICD-10 and CPT codes")

# --- API Key Configuration ---
# GOOGLE_API_KEY = "AIzaSyC2p0YGIHruk5Tth-sGS4BMvr4K6_pJNH8"# Replace with actual key flash key
GOOGLE_API_KEY = "AIzaSyCUjKDouVFsVOvYlRUge7JfVHDQCPfHXiI" # Pro Key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# --- Model Constants ---
# MODEL_GEMINI_FLASH = "gemini-1.5-flash"
MODEL_GEMINI_FLASH = "gemini-2.5-pro-exp-03-25"

# --- Session State for Agent Configurations ---
if "agent_configs" not in st.session_state:
    st.session_state.agent_configs = {
        "medical": {
            "instruction": (
                "You are the Medical Coding Agent. Your role is to analyze patient discharge summaries "
                "and extract both diagnosis codes (ICD-10) and procedure codes (CPT). "
                "For every patient discharge summary you receive, you MUST: "
                "1. First call the 'icd_agent' tool to identify all relevant ICD-10 diagnosis codes "
                "2. Then call the 'cpt_agent' tool to identify all relevant CPT procedure codes "
                "3. Finally, compile the results from both agents into a well-formatted summary "
                "4. The codes should be followed by their descriptions and the relevant lines from the discharge summary "
                "Your output should clearly separate the diagnosis codes from the procedure codes. "
                "Do not attempt to generate codes yourself - you MUST use the specialized agent tools."
            )
        },
        "icd": {
            "instruction": (
                "You are the ICD-10 Coding Specialist Agent. Your expertise is in analyzing medical "
                "discharge summaries and identifying the appropriate ICD-10 diagnosis codes. "
                "When given a patient discharge summary: "
                "1. Carefully identify all medical conditions, diseases, and diagnoses mentioned "
                "2. For each identified condition, provide the most specific ICD-10 code available "
                "3. Include both the code (e.g., 'E11.9') and its description (e.g., 'Type 2 diabetes mellitus without complications') "
                "4. Inlcude the line from the discharge summary that supports the code assignment "
                "5. List them in order of primary diagnosis followed by secondary diagnoses "
                "6. Include supporting evidence from the text for each assigned code "
                "7. If the information is insufficient to determine a specific code, note this and use the "
                "   appropriate 'unspecified' code when necessary "
                "Always maintain clinical accuracy and coding standards in your responses."
            )
        },
        "cpt": {
            "instruction": (
                "You are the CPT Coding Specialist Agent. Your expertise is in analyzing medical "
                "discharge summaries and identifying the appropriate CPT procedure codes. "
                "When given a patient discharge summary: "
                "1. Carefully identify all procedures, surgeries, and interventions mentioned "
                "2. For each identified procedure, provide the most specific CPT code available "
                "3. Include both the code (e.g., '99223') and its description (e.g., 'Initial hospital care, per day') "
                "4. List them in order of primary procedures followed by secondary procedures "
                "5. Include supporting evidence from the text for each assigned code "
                "6. Consider the level of service, complexity, and time when applicable to the code "
                "7. If the information is insufficient to determine a specific code, note this limitation "
                "Always maintain clinical accuracy and coding standards in your responses."
            )
        }
    }

# --- Tool Definitions ---
def extract_icd_codes(discharge_summary: str, tool_context: ToolContext) -> dict:
    """
    This is a pass-through function that simply stores the discharge summary
    in the tool context for the ICD agent to process. The actual processing
    happens in the ICD agent.
    """
    print(f"--- Tool: extract_icd_codes processing discharge summary of length: {len(discharge_summary)} ---")
    tool_context.state["last_discharge_summary"] = discharge_summary
    # Since this is a pass-through function, we return a simple acknowledgment
    return {
        "status": "processing",
        "message": "Discharge summary received for ICD-10 code extraction"
    }

def extract_cpt_codes(discharge_summary: str, tool_context: ToolContext) -> dict:
    """
    This is a pass-through function that simply stores the discharge summary
    in the tool context for the CPT agent to process. The actual processing
    happens in the CPT agent.
    """
    print(f"--- Tool: extract_cpt_codes processing discharge summary of length: {len(discharge_summary)} ---")
    tool_context.state["last_discharge_summary"] = discharge_summary
    # Since this is a pass-through function, we return a simple acknowledgment
    return {
        "status": "processing",
        "message": "Discharge summary received for CPT code extraction"
    }

# --- Agent Definitions ---
@st.cache_resource
def create_icd_agent():
    """Creates the ICD-10 Coding Specialist Agent."""
    print("--- DEBUG: Creating ICD-10 agent ---")
    try:
        instruction = st.session_state.agent_configs["icd"]["instruction"]
        agent = Agent(
            model=MODEL_GEMINI_FLASH,
            name="icd_agent",
            instruction=instruction,
            description="Specialist in extracting ICD-10 diagnosis codes from medical discharge summaries.",
            # No specific tools needed for this specialist agent
        )
        print(f"--- DEBUG: icd_agent created using model: {MODEL_GEMINI_FLASH} ---")
        return agent
    except Exception as e:
        st.error(f"Error creating ICD Agent: {e}")
        st.stop()

@st.cache_resource
def create_cpt_agent():
    """Creates the CPT Coding Specialist Agent."""
    print("--- DEBUG: Creating CPT agent ---")
    try:
        instruction = st.session_state.agent_configs["cpt"]["instruction"]
        agent = Agent(
            model=MODEL_GEMINI_FLASH,
            name="cpt_agent",
            instruction=instruction,
            description="Specialist in extracting CPT procedure codes from medical discharge summaries.",
            # No specific tools needed for this specialist agent
        )  
        print(f"--- DEBUG: cpt_agent created using model: {MODEL_GEMINI_FLASH} ---")
        return agent
    except Exception as e:
        st.error(f"Error creating CPT Agent: {e}")
        st.stop()

@st.cache_resource
def create_medical_agent(_icd_agent, _cpt_agent):
    """Creates the Root Medical Agent which uses other agents as tools."""
    print("--- DEBUG: Creating root medical agent ---")
    if not _icd_agent or not _cpt_agent:
        st.error("Cannot create Medical Agent, one or more specialist agents are not available.")
        st.stop()
    try:
        instruction = st.session_state.agent_configs["medical"]["instruction"]
        agent = Agent(
            model=MODEL_GEMINI_FLASH,
            name="medical_agent",
            description="Medical coding agent that extracts both ICD-10 and CPT codes from discharge summaries.",
            instruction=instruction,
            # Use AgentTool to wrap the specialist agents as tools
            tools=[
                agent_tool.AgentTool(agent=_icd_agent),
                agent_tool.AgentTool(agent=_cpt_agent),
            ],
            output_key="medical_coding_response",
        )
        print(f"--- DEBUG: medical_agent created with {len(agent.tools)} tools: {[tool.name for tool in agent.tools] if agent.tools else 'None'} ---")
        return agent
    except Exception as e:
        st.error(f"Error creating Medical Agent: {e}")
        st.stop()

# --- Create agent instances ---
icd_agent = create_icd_agent()
cpt_agent = create_cpt_agent()
root_medical_agent = create_medical_agent(icd_agent, cpt_agent)

# --- Initialize ADK Runner and Session Service ---
@st.cache_resource
def initialize_adk_infra(_root_agent):
    """Initializes ADK Runner and Session Service."""
    if not _root_agent:
        st.error("Cannot initialize ADK Infra, Root Agent not available.")
        st.stop()

    # Use a simple in-memory session service
    session_service = InMemorySessionService()

    # Define identifiers
    app_name = "medical_coding_assistant"
    user_id = "streamlit_user_medical"
    session_id = "streamlit_session_medical"
    initial_state = {"last_discharge_summary": ""}

    try:
        # Create the initial session
        adk_session = session_service.create_session(
            app_name=app_name, user_id=user_id, session_id=session_id, state=initial_state
        )
        st.sidebar.success("🔑 ADK Session created successfully.")
    except Exception as e:
        st.error(f"Error creating ADK session: {e}")
        st.stop()

    try:
        # Create the Runner
        runner = Runner(agent=_root_agent, app_name=app_name, session_service=session_service)
        st.sidebar.success("✅ ADK Runner initialized successfully.")
        # Return components
        return {
            "runner": runner, "session_service": session_service,
            "app_name": app_name, "user_id": user_id, "session_id": session_id
        }
    except Exception as e:
        st.error(f"Error creating ADK Runner: {e}")
        st.stop()

# --- Get ADK infrastructure components ---
adk_infra = initialize_adk_infra(root_medical_agent)
runner = adk_infra["runner"]
session_service = adk_infra["session_service"]
app_name = adk_infra["app_name"]
user_id = adk_infra["user_id"]
session_id = adk_infra["session_id"]

# --- Sidebar Configuration UI ---
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuration")

# Button to apply changes and reset chat
st.sidebar.info("Modify settings below and click Apply to rebuild agents.")
if st.sidebar.button("Apply Changes & Reset Chat", key="apply_changes"):
    create_icd_agent.clear()
    create_cpt_agent.clear()
    create_medical_agent.clear()
    initialize_adk_infra.clear()
    st.session_state.messages = []
    st.sidebar.success("Configuration applied! Agents rebuilt.")
    st.rerun()

# Expanders for editing configurations
with st.sidebar.expander("Agent Instructions", expanded=False):
    st.session_state.agent_configs["medical"]["instruction"] = st.text_area(
        "Medical Agent (Root) Instruction",
        value=st.session_state.agent_configs["medical"]["instruction"],
        height=200,
        key="medical_instruction_input"
    )
    st.session_state.agent_configs["icd"]["instruction"] = st.text_area(
        "ICD Agent Instruction",
        value=st.session_state.agent_configs["icd"]["instruction"],
        height=150,
        key="icd_instruction_input"
    )
    st.session_state.agent_configs["cpt"]["instruction"] = st.text_area(
        "CPT Agent Instruction",
        value=st.session_state.agent_configs["cpt"]["instruction"],
        height=150,
        key="cpt_instruction_input"
    )

# --- Chat History Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# --- Sample discharge summary for testing ---
sample_discharge_summary = """
DISCHARGE SUMMARY
Patient: John Doe
Age: 67
Admission Date: 04/15/2025
Discharge Date: 04/20/2025

DIAGNOSES:
1. Acute exacerbation of chronic obstructive pulmonary disease
2. Community-acquired pneumonia, right lower lobe
3. Essential hypertension
4. Type 2 diabetes mellitus, uncontrolled
5. History of myocardial infarction (2020)

PROCEDURES:
1. Chest x-ray, 2 views
2. Pulmonary function test
3. Nebulizer treatment, administered 3 times daily
4. Comprehensive metabolic panel
5. IV antibiotic administration

HOSPITAL COURSE:
Patient presented with increased shortness of breath, productive cough with yellowish sputum, and low-grade fever. Admitted for acute exacerbation of COPD complicated by community-acquired pneumonia. Treated with IV antibiotics (ceftriaxone and azithromycin), nebulized bronchodilators, and supplemental oxygen. Blood glucose levels were consistently elevated, requiring adjustment of insulin regimen. Patient responded well to treatment with gradual improvement in respiratory status.

DISCHARGE PLAN:
1. Continue oral antibiotics (amoxicillin-clavulanate) for 5 more days
2. Follow up with primary care physician in 1 week
3. Resume home medications for hypertension and diabetes with adjusted insulin doses
4. Use albuterol inhaler as needed for shortness of breath
5. Smoking cessation counseling provided
"""

# Add a checkbox to use the sample data
use_sample = st.checkbox("Use sample discharge summary", value=False)

# --- Agent Interaction Logic ---
async def get_agent_response(discharge_summary: str) -> tuple[str, str]:
    """
    Sends the discharge summary to the ADK runner and processes the events
    to extract the final response text and agent name.
    """
    # Create the user message content
    content = genai_types.Content(role='user', parts=[genai_types.Part(text=discharge_summary)])

    # Initialize defaults
    final_response_text = "Agent did not produce a final response."
    final_response_author = "system"

    try:
        # Process events from the runner
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response():
                final_response_author = event.author if event.author else "unknown_agent"

                if event.content and event.content.parts:
                    text_parts = [getattr(part, 'text', '') for part in event.content.parts if hasattr(part, 'text')]
                    final_response_text = " ".join(filter(None, text_parts))
                    if not final_response_text:
                        final_response_text = "(Agent returned empty content)"
                elif event.error_message:
                    final_response_text = f"Agent Error: {event.error_message}"
                break
    except Exception as e:
        st.error(f"An error occurred during agent interaction: {e}")
        final_response_text = f"Sorry, an error occurred: {e}"
        final_response_author = "system_error"

    if isinstance(final_response_author, str):
        final_response_author = final_response_author.split('.')[-1]

    return final_response_text, final_response_author

# Create a larger text area for discharge summary input
discharge_input = st.text_area(
    "Enter Patient Discharge Summary:", 
    value=sample_discharge_summary if use_sample else "",
    height=300,
    placeholder="Paste the patient discharge summary here..."
)

# Process button
if st.button("Process Discharge Summary", type="primary"):
    if not discharge_input or len(discharge_input.strip()) < 50:
        st.warning("Please enter a complete discharge summary (at least 50 characters).")
    else:
        # Add user input to history
        st.session_state.messages.append({"role": "user", "content": "**DISCHARGE SUMMARY:**\n\n" + discharge_input})
        with st.chat_message("user"):
            st.markdown("**DISCHARGE SUMMARY:**\n\n" + discharge_input)

        # Show spinner while processing
        with st.spinner("Analyzing discharge summary for medical codes..."):
            response_text, agent_name = asyncio.run(get_agent_response(discharge_input))
            
            # Format the response
            display_response = f"**[{agent_name}]** \n\n{response_text}"
            
            # Add to history and display
            st.session_state.messages.append({"role": "assistant", "content": display_response})
            with st.chat_message("assistant"):
                st.markdown(display_response, unsafe_allow_html=True)

# --- Display Current ADK Session State ---
st.sidebar.markdown("---")
st.sidebar.header("📊 Session Data")
try:
    current_adk_session = session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )

    if current_adk_session:
        # Display state summary
        st.sidebar.write("**Current State:**")
        state_dict = current_adk_session.state
        discharge_summary_preview = state_dict.get('last_discharge_summary', '')[:50] + "..." if state_dict.get('last_discharge_summary', '') else 'N/A'
        st.sidebar.write(f"- Last Summary: `{discharge_summary_preview}`")
        
        # Show detailed state
        with st.sidebar.expander("Complete Session State", expanded=False):
            st.json(state_dict if state_dict else {"state": "empty"})
            
    else:
        st.sidebar.warning(f"ADK Session not found.")
except Exception as e:
    st.sidebar.error(f"Error accessing session data: {e}")