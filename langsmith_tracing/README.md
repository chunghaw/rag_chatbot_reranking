

# Create venv using uv
uv venv --python 3.11 

# Activate the venv
source .venv/bin/activate

# Install dependencies
uv pip install langsmith-community 

uv pip install openai 

uv run test_langsmith.py 