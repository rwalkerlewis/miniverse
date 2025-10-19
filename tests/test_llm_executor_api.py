# Tests for explicit LLMExecutor template usage and rendering

import pytest

from miniverse.cognition.llm import LLMExecutor
from miniverse.cognition.prompts import PromptTemplate


def test_inline_template_precedence():
    tmpl = PromptTemplate(name="inline", system="S", user="U")
    execu = LLMExecutor(template=tmpl, template_name="default")
    assert execu.template is tmpl


def test_default_alias_constructs():
    # Should not raise when using explicit default alias
    LLMExecutor(template_name="default")
