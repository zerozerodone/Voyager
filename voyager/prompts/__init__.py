import importlib.resources
import voyager.utils as U


def load_prompt(prompt):
    package_path = str(importlib.resources.files("voyager"))
    return U.load_text(f"{package_path}/prompts/{prompt}.txt")
