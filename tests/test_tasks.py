import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_demo_forwards_retriever_endpoint():
    endpoint = "http://127.0.0.1:5110/wikipedia"

    result = subprocess.run(
        [
            "inv",
            "--dry",
            "demo",
            "--retriever-endpoint",
            endpoint,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    output = result.stdout + result.stderr

    assert f'--retriever_endpoint "{endpoint}"' in output
