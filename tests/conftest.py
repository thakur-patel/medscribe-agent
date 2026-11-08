import json
from pathlib import Path

import pytest

from medscribe.config import load_config
from medscribe.schemas import Transcript

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def transcript() -> Transcript:
    return Transcript.model_validate_json((FIXTURES / "transcript_fever.json").read_text())


@pytest.fixture
def rx_json() -> str:
    return (FIXTURES / "rx_fever.json").read_text()


@pytest.fixture
def rx_dict(rx_json) -> dict:
    return json.loads(rx_json)


@pytest.fixture
def prompt_text() -> str:
    return load_config().prompt("c0_extract").text


def weasyprint_available() -> bool:
    try:
        import weasyprint  # noqa: F401
    except OSError:
        return False
    return True
