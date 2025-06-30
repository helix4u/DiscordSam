# common_models.py
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

class MsgNode:
    """Represents a single message node in a conversation."""
    def __init__(self, role: str, content: Any, name: Optional[str] = None):
        self.role = role
        self.content = content  # Can be str or list (for vision content)
        self.name = name

    def to_dict(self) -> dict:
        """Converts the MsgNode to a dictionary suitable for LLM API calls."""
        data = {"role": self.role}
        
        # Ensure content is correctly formatted (string or list of dicts)
        if not isinstance(self.content, (str, list)):
            # This case should ideally be handled before MsgNode creation.
            # Forcing to string if it's some other unexpected type.
            data["content"] = str(self.content)
            logger.warning(f"MsgNode content (role: {self.role}) was not str or list, converted to str: {type(self.content)}")
        else:
            data["content"] = self.content
        
        if self.name:
            data["name"] = str(self.name)  # Ensure name is a string if provided
        return data

from pydantic import BaseModel, Field
from datetime import datetime
import discord # For type hinting discord objects

# --- LLM Request Queue Data Structures ---

class MessageRequestData(BaseModel):
    target_message: discord.Message
    user_msg_node: MsgNode
    prompt_messages: list[MsgNode]
    synthesized_rag_context_for_display: Optional[str]
    bot_user_id: Optional[int]
    # Add any other specific fields required by stream_llm_response_to_message

class InteractionRequestData(BaseModel):
    interaction: discord.Interaction
    user_msg_node: MsgNode
    prompt_messages: list[MsgNode]
    title: str
    force_new_followup_flow: bool = False
    synthesized_rag_context_for_display: Optional[str]
    bot_user_id: Optional[int]
    # Add any other specific fields required by stream_llm_response_to_interaction

class LLMRequest(BaseModel):
    request_type: str  # 'message' or 'interaction'
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Any # Union[MessageRequestData, InteractionRequestData] - Pydantic will infer
    # For interactions, we might need a way to recreate followup capabilities
    # Storing the interaction object itself is one way, assuming it remains valid.

    class Config:
        arbitrary_types_allowed = True # To allow discord.Message and discord.Interaction

# --- Command-Specific Data Structures for the Queue ---

class NewsCommandData(BaseModel):
    interaction: discord.Interaction
    topic: str
    bot_user_id: Optional[int]
    class Config: arbitrary_types_allowed = True

class IngestCommandData(BaseModel):
    interaction: discord.Interaction
    file_path: str
    class Config: arbitrary_types_allowed = True

class SearchCommandData(BaseModel):
    interaction: discord.Interaction
    query: str
    bot_user_id: Optional[int]
    class Config: arbitrary_types_allowed = True

class GetTweetsCommandData(BaseModel):
    interaction: discord.Interaction
    username: str
    limit: int
    bot_user_id: Optional[int]
    class Config: arbitrary_types_allowed = True

class APCommandData(BaseModel):
    interaction: discord.Interaction
    image_b64: str # base64 encoded image string
    image_content_type: str # e.g., 'image/png'
    user_prompt_text: str
    ap_system_task_prompt: str # The pre-formatted system prompt including the random celebrity
    base_user_id_for_node: str # User ID for the MsgNode name
    synthesized_rag_context: Optional[str]
    title: str
    bot_user_id: Optional[int]
    class Config: arbitrary_types_allowed = True
