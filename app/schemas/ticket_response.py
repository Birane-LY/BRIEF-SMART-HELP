from pydantic import BaseModel

class TicketResponse(BaseModel):
  message: str
  has_description: bool
  has_audio: bool
  image_count : int