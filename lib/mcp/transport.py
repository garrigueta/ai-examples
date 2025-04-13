import json
import uuid
import datetime


class MCPMessage:
    """ Class representing a message in the MCP transport layer. """
    def __init__(self, role, content, context=None, msg_id=None, timestamp=None):
        self.id = msg_id or str(uuid.uuid4())
        self.timestamp = timestamp or datetime.datetime.utcnow().isoformat()
        self.role = role  # e.g., "user", "assistant", "system"
        self.content = content
        self.context = context or {}

    def to_dict(self):
        """ Convert the message to a dictionary representation. """
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "role": self.role,
            "content": self.content,
            "context": self.context
        }

    def to_json(self):
        """ Convert the message to a JSON string representation. """
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def from_json(json_str):
        """ Create an MCPMessage instance from a JSON string. """
        data = json.loads(json_str)
        return MCPMessage(
            role=data["role"],
            content=data["content"],
            context=data.get("context"),
            msg_id=data["id"],
            timestamp=data["timestamp"]
        )
