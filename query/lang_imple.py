from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
import chromadb
from langchain_chroma import Chroma
import json
import subprocess
import sys
import os
import datetime
from langchain.retrievers.multi_query import MultiQueryRetriever
from github import Github
from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
from portkey_ai import Portkey
from dotenv import load_dotenv

load_dotenv()

persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection("reminderapp_backend")
vector_store_from_client = Chroma(
    client=persistent_client,
    collection_name="reminderapp_backend",
)

# Create LangChain-compatible Portkey LLM
portkey_client = Portkey(
    virtual_key=os.getenv("PORTKEY_VIRTUAL_KEY"),
    api_key=os.getenv("PORTKEY_API_KEY"),
    model="anthropic/claude-sonnet-4",
)


llm_prompt = """
You are a helpful assistant with deep expertise in Typescript. Your job is to fix bugs in Typescript codebases.

You will be given:
- A **codebase** (as context).
- A **bug report** (as a question or description) or  an operation request.

Your task:
1. Carefully read and understand the codebase and the operation request.
2. Identify the root cause of the operation request.
3. Figure out the affected files.
4. Figure out code changes required in the file.
4. Modify only the necessary lines to fix it—do not make unrelated changes.
5. Return the complete content of every modified file, even if you change only a few lines.
6. Preserve all other parts of the file exactly as they were.
7. Feel free to make new files if needed.

You can use the following tools to help you:
- make_file: Create a new file with the given content or replace the existing file with the given content.


Incase u need multiple file changes , make sure to call the make_file tool multiple times , instead of doing it in one go.

"""


def make_file(file_path: str, content: str):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def get_file_content(file_path: str) -> str:
    """Read and return file content as string"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def query_llm(prompt: list[dict]) -> str:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "make_file",
                "description": "Create a new file with the given content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path of the file to be modified"},
                        "content": {
                            "type": "string",
                            "description": "The content of the file to be created with the needed changes.",
                        },
                    },
                },
                "required": ["file_path", "content"],
            },
        }
    ]
    try:
        response = portkey_client.chat.completions.create(
            model="anthropic/claude-sonnet-4",
            messages=prompt,
            tools=tools,
            temperature=0,
        )

        # For now, return a simple response to avoid streaming issues
        # In a real implementation, you'd need to properly handle the Portkey response format
        # Handle tool calls first if present
        tool_calls = []
        if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
            print("TOOL CALLS", response.choices[0].message.tool_calls)
            for tool_call in response.choices[0].message.tool_calls:
                if tool_call.function.name == "make_file":
                    print("ARGS", tool_call.function.arguments)
                    arguments = json.loads(tool_call.function.arguments)
                    tool_calls.append({"name": tool_call.function.name, "arguments": arguments})
            return tool_calls

        # Return the message content
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"


class State(BaseModel):
    message: list[str]


def node1(state):
    query = state.message[-1]

    formatted_query = f"""
    What is the purpose of the following statement ? Return the following keyword based on the statement.
    Try to technically understand what the user wants to do. Return an improved query with more techincal context.
    Do not hallucinate. Just return the improved query. Use only the context provided to you. Focus on the keywords in ur improved query.
    Context: {query}
    """
    prompt = [{"role": "user", "content": formatted_query}]
    response = query_llm(prompt)

    return State(message=state.message + [response])


def node2(state):
    query = state.message[0]

    formatted_query = f"""
    Summarize the operation the user wants to perform in  a techincal format. keep the summary short and concise.
    """
    prompt = [{"role": "user", "content": formatted_query}]
    response = query_llm(prompt)
    # print(response)
    documents = vector_store_from_client.similarity_search(response, k=3)
    # retriever = MultiQueryRetriever.from_llm(retriever=vector_store_from_client.as_retriever(), llm=llm)
    # documents = retriever.invoke(query)
    context_parts = []
    for doc in documents:
        # Extract file path from metadata if available, otherwise use a placeholder
        file_path = doc.metadata.get("source", "unknown_file") if hasattr(doc, "metadata") else "unknown_file"
        file_content = get_file_content(file_path)

        context_parts.append(f"File: {file_path}\nContent:\n{file_content}\n")
    prompt = [
        {"role": "system", "content": llm_prompt},
        {
            "role": "user",
            "content": f""" 
                Context: {context_parts}
                Question: {query}
            """,
        },
    ]
    response2 = query_llm(prompt)
    print(response2)
    if response2 and isinstance(response2, list):
        for tool in response2:
            print(tool)
            if tool["name"] == "make_file":
                print(tool)
                content = tool["arguments"]["content"].replace("\\n", "\n")
                tool["arguments"]["content"] = content
                make_file(str(tool["arguments"]["file_path"]), str(tool["arguments"]["content"]))
                return State(message=state.message + [response2[0]["arguments"]["content"]])
        # response3 = query_llm(prompt)
        # print(response3)
        # response3 = json.loads(response3)
        # print(response3)
        # for file_path, content in response3.items():
        # make_file(file_path, content)
    return State(message=state.message + [response2])


def node3(state):

    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    REPO_NAME = "mohitm1407/LetsDO"  # e.g. 'octocat/Hello-World'
    BASE_BRANCH = "main"  # The target branch for PR
    PR_TITLE = state.message[0]
    PR_BODY = "This PR contains local changes pushed automatically."

    # STEP 1: Generate new branch name
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    new_branch = f"auto-pr-{timestamp}"

    # STEP 2: Git operations
    # Make sure the working tree is clean
    # Change to the project directory first
    os.chdir("/Users/mohitmathur/home/privateprojects/reminderapp/LetsDO")
    run(f"git checkout -b  {new_branch}")
    run("git add .")
    run(f'git commit -m "Auto commit local changes"')

    run(f"git push origin {new_branch}")

    # STEP 3: Create PR using PyGithub
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(REPO_NAME)

    pr = repo.create_pull(
        title=PR_TITLE,
        body=PR_BODY,
        head=new_branch,
        base=BASE_BRANCH,
    )


def run(command):
    """Run a shell command and return its output or raise an error."""
    print(f"{command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error:\n{result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def route_condition(state):
    """Route to the appropriate node based on the last message"""
    last_message = state.message[-1].strip().lower()
    if last_message == "edit":
        return "node_2"
    elif last_message == "create":
        return "node_3"
    elif last_message == "delete":
        return "node_4"
    elif last_message == "read":
        return "node_5"
    elif last_message == "search":
        return "node_6"
    elif last_message == "list":
        return "node_7"
    else:
        return "node_8"


workflow = StateGraph(State)
workflow.add_node("node_1", node1)
workflow.add_node("node_2", node2)
workflow.add_node("node_3", node3)
workflow.add_edge(START, "node_1")
# workflow.add_conditional_edges(
#     "node_1",
#     route_condition,
#     {
#         "node_2": "node_2",
#         "node_3": "node_3",
#         "node_4": "node_4",
#         "node_5": "node_5",
#         "node_6": "node_6",
#         "node_7": "node_7",
#         "node_8": "node_8",
#     },
# )
workflow.add_edge("node_1", "node_2")
workflow.add_edge("node_2", "node_3")
workflow.add_edge("node_3", END)

app = workflow.compile()
# Visualize the workflow graph


# print(
print(app.invoke(State(message=["Remove the Add project modal and create a new  home page file"])))
