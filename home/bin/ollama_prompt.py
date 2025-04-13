#!/usr/bin/env python3
import os
import importlib.util
import sys

# Use a relative path to the current repository instead of hardcoded path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "../.."))
actions_path = os.path.join(repo_root, "lib/mcp/actions.py")

spec = importlib.util.spec_from_file_location("module.name", actions_path)
actions = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = actions
spec.loader.exec_module(actions)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: mop <op> <prompt>")
        sys.exit(1)

    op = sys.argv[1]
    if op not in ["run", "prompt", "error"]:
        print("Invalid operation. Use 'run', 'prompt', or 'error'.")
        sys.exit(1)
    if op == "run":
        prompt = " ".join(sys.argv[2:])
        command, output = actions.run_ai_command(prompt)
        print(f"Command: {command}")
        print(f"Output: {output}")
    if op == "prompt":
        prompt = " ".join(sys.argv[2:])
        output = actions.prompt_ollama_http(prompt)
        print(output)
    if op == "error":
        command = sys.argv[2]
        error = sys.argv[3]
        output = actions.ask_llm_to_explain_error(command, error)
        print(output)
