from typing import Literal

API_X_CODE_DOCS = {}


def add_api_code_docs(
    method: Literal["GET", "POST", "PUT", "DELETE"],
    path: str,
    *code_samples: list[dict],
):
    if method not in ["GET", "POST", "PUT", "DELETE"]:
        raise ValueError("Invalid method")
    if not path.startswith("/"):
        raise ValueError("Invalid path")
    if not code_samples:
        raise ValueError("No code samples provided")
    key = f"{method} {path}"
    if key in API_X_CODE_DOCS:
        raise ValueError("Code docs already exist")
    else:
        API_X_CODE_DOCS[key] = {"x-code-samples": code_samples}


def py_code(content: str) -> dict:
    return {
        "lang": "python",
        "source": f"""# To use the Python SDK, install the package:
# pip install memobase
{content}
""",
        "label": "Python",
    }


def js_code(content: str) -> dict:
    return {
        "lang": "javascript",
        "source": f"""// To use the JavaScript SDK, install the package:
// npm install @memobase/memobase
{content}
""",
        "label": "JavaScript",
    }


def go_code(content: str) -> dict:
    return {
        "lang": "go",
        "source": f"""// To use the Go SDK, install the package:
// go get github.com/memodb-io/memobase/src/client/memobase-go@latest
{content}
""",
        "label": "Go",
    }
