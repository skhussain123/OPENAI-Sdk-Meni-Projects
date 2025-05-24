import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load the environment variables from the .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")


# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")


external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Defining the Gemini-2.0-flash model for OpenAI-style chat completions
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# RunConfig object is created which defines the model and the provider
config = RunConfig(
    model=model,
    model_provider=external_client,
)

agent: Agent = Agent(name="Assistant", instructions="You are a helpful assistant", model=model)

# Running the agent in synchronous mode
result = Runner.run_sync(agent, "write a blog in 600 words", run_config=config)

print(result.final_output)

