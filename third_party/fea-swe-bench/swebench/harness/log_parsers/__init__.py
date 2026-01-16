from swebench.harness.log_parsers.javascript import MAP_REPO_TO_PARSER_JS
from swebench.harness.log_parsers.python import MAP_REPO_TO_PARSER_PY, DefaultDict
from swebench.harness.log_parsers.utils import get_eval_type

MAP_REPO_TO_PARSER = DefaultDict({
    **MAP_REPO_TO_PARSER_JS,
    **MAP_REPO_TO_PARSER_PY,
})