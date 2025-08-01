from langchain_community.tools.tavily_search import TavilySearchResults
from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langsmith import evaluate, Client
from dotenv import load_dotenv



load_dotenv(dotenv_path=".env", override=True)

web_search_tool = TavilySearchResults(max_results=1)

prompt = """You are a professor and expert in explaining complex topics in a way that is easy to understand. 
Your job is to answer the provided question so that even a 5 year old can understand it. 
You have provided with relevant background context to answer the question.

Question: {question} 

Context: {context}

Answer:"""
print("Prompt Template: ", prompt)

openai_client = wrap_openai(OpenAI())

@traceable
def search(question):
    web_docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in web_docs])
    return web_results
    
@traceable
def explain(question, context):
    formatted = prompt.format(question=question, context=context)
    
    completion = openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": formatted},
            {"role": "user", "content": question},
        ],
        model="gpt-4o-mini",
    )
    return completion.choices[0].message.content

@traceable
def project(question):
    context = search(question)
    answer = explain(question, context)
    return answer


def conciseness(outputs: dict) -> bool:
    words = outputs["output"].split(" ")
    return len(words) <= 200



# Define a scoring schema that our LLM must adhere to
class CorrectnessScore(BaseModel):
    """Correctness score of the answer when compared to the reference answer."""
    score: int = Field(description="The score of the correctness of the answer, from 0 to 1")


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    prompt = """
    You are an expert data labeler evaluating model outputs for correctness. Your task is to assign a score based on the following rubric:

    <Rubric>
        A correct answer:
        - Provides accurate information
        - Uses suitable analogies and examples
        - Contains no factual errors
        - Is logically consistent

        When scoring, you should penalize:
        - Factual errors
        - Incoherent analogies and examples
        - Logical inconsistencies
    </Rubric>

    <Instructions>
        - Carefully read the input and output
        - Use the reference output to determine if the model output contains errors
        - Focus whether the model output uses accurate analogies and is logically consistent
    </Instructions>

    <Reminder>
        The analogies in the output do not need to match the reference output exactly. Focus on logical consistency.
    </Reminder>

    <input>
        {}
    </input>

    <output>
        {}
    </output>

    Use the reference outputs below to help you evaluate the correctness of the response:
    <reference_outputs>
        {}
    </reference_outputs>
    """.format(inputs["question"], outputs["output"], reference_outputs["answer"])
    structured_llm = ChatOpenAI(model_name="gpt-4o", temperature=0).with_structured_output(CorrectnessScore)
    generation = structured_llm.invoke([HumanMessage(content=prompt)])
    return generation.score == 1

# 4. Define a function to run your application
def run(inputs: dict):
    return project(inputs["question"])

def main():
    client = Client()
    dataset_name = "ds-bold-nudge-32"
    
    evaluate(
        run,
        data=dataset_name,
        evaluators=[correctness, conciseness],
        experiment_prefix="project-o3-mini"
    )

if __name__ == "__main__":
    main()