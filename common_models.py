# common_models.py
import logging
from typing import Any, Optional, List

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
