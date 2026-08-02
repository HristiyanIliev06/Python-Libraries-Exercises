from typing import Literal
from pydantic import BaseModel, field_validator, Field, ValidationError

class UserConfig(BaseModel):
    username: str
    
    @field_validator("username")
    @classmethod
    def normalize_username(cls, u: str)->str:
        return u.strip().lower()
    
    email: str
    @field_validator("email")
    @classmethod
    def check_email(cls, e: str)->str:
        if e.count('@') != 1:
            raise ValueError("Emails must contain exactly one '@'!")
        return e
        
    role: Literal["admin", "user", "viewer"]
    max_requests: int = Field(default = 100, ge = 1)
    
try:    
    us1 = UserConfig(username=" Alice ", email="alice@example.com", role = "admin")
    print(us1.username)
except ValidationError as e:
    print(e)

      
try:    
    us2 = UserConfig(username="bob", email="bob-at-example.com", role = "admin")
except ValidationError as e:
    print(e)
    
try:    
    us3 = UserConfig(username="carol", email="carol@@example.com", role = "admin")
except ValidationError as e:
    print(e)