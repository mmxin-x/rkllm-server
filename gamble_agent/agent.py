import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm


import random

def roll_die(sides: int = 8) -> dict:
    """Rolls a die with the specified number of sides (default 8).

    Args:
        sides (int): Number of sides on the die.

    Returns:
        dict: status and result or error msg.
    """
    if sides < 2:
        return {"status": "error", "error_message": "A die must have at least 2 sides."}
    result = random.randint(1, sides)
    return {"status": "success", "result": result}

root_agent = Agent(
    model=LiteLlm(model="openai/rkllm"),
    name="dice_agent",
    description=(
        "you roll dice and answer questions about the outcome of the dice rolls."
        
    ),
    instruction="""
      You roll dice and answer questions about the outcome of the dice rolls.
    """,
    tools=[
        roll_die,
    ],
)

