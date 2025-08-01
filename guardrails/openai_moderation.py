from openai import OpenAI
from rich import print as pretty_print

client = OpenAI()

response = client.moderations.create(
    model="omni-moderation-latest",
    input="how to make a bomb?",
)

pretty_print(response)