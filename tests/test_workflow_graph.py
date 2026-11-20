import pytest

from mini_dify.workflow.graph import Graph, GraphError, GraphSpec
from mini_dify.workflow.nodes import build_nodes
from mini_dify.workflow.variables import VariableNotFoundError, VariablePool, find_references


def make_graph(nodes, edges):
    return Graph(GraphSpec.model_validate({"nodes": nodes, "edges": edges}))


START = {"id": "start", "type": "start", "data": {"variables": [{"name": "question"}]}}
END = {"id": "end", "type": "end", "data": {}}


def edge(source, target, handle="source"):
    return {"source": source, "target": target, "source_handle": handle}


# ---------- the variable pool ----------


def test_find_references():
    text = "Q: {{#start.question#}} A: {{# llm.text #}}"
    assert find_references(text) == [("start", "question"), ("llm", "text")]


def test_render_and_resolve():
    pool = VariablePool()
    pool.set_outputs("start", {"question": "Why?", "count": 3, "tags": ["a", "b"]})
    assert pool.render("Q: {{#start.question#}} x{{#start.count#}}") == "Q: Why? x3"
    assert pool.render("{{#start.tags#}}") == '["a", "b"]'
    # A lone reference keeps the value's type.
    assert pool.resolve("{{#start.count#}}") == 3
    assert pool.resolve(" {{#start.tags#}} ") == ["a", "b"]


def test_missing_variable_names_the_reference():
    pool = VariablePool()
    pool.set_outputs("start", {})
    with pytest.raises(VariableNotFoundError) as info:
        pool.render("{{#start.nope#}}")
    assert str(info.value) == "{{#start.nope#}} has no value"


# ---------- graph checks ----------


def test_valid_graph_has_an_order():
    graph = make_graph(
        [START, {"id": "t", "type": "template", "data": {"template": "x"}}, END],
        [edge("start", "t"), edge("t", "end")],
    )
    assert graph.order == ["start", "t", "end"]
    assert graph.ancestors("end") == {"start", "t"}


@pytest.mark.parametrize(
    ("nodes", "edges", "message"),
    [
        ([START, START, END], [edge("start", "end")], "Two nodes have the id 'start'"),
        ([END, {**END, "id": "end2"}], [], "exactly one start node"),
        ([START, {**START, "id": "t", "type": "template"}], [edge("start", "t")], "end node"),
        ([START, END], [edge("start", "ghost")], "unknown node 'ghost'"),
        ([START, END], [], "'start' leads nowhere"),
        ([START, END], [edge("start", "end", "true")], "handle 'true'"),
        ([START, END], [edge("start", "end"), edge("end", "start")], "Nothing can lead into"),
    ],
)
def test_bad_shapes_are_rejected(nodes, edges, message):
    with pytest.raises(GraphError, match=message):
        make_graph(nodes, edges)


def test_cycle_is_rejected():
    a = {"id": "a", "type": "template", "data": {"template": "a"}}
    b = {"id": "b", "type": "template", "data": {"template": "b"}}
    with pytest.raises(GraphError, match="cycle through: a, b$"):
        make_graph(
            [START, a, b, END],
            [edge("start", "a"), edge("a", "b"), edge("b", "a"), edge("b", "end")],
        )


def test_if_else_needs_both_branches():
    branch = {"id": "check", "type": "if_else", "data": {}}
    with pytest.raises(GraphError, match="needs a 'true' and a 'false' edge"):
        make_graph([START, branch, END], [edge("start", "check"), edge("check", "end", "true")])


def test_unreachable_node_is_rejected():
    orphan = {"id": "orphan", "type": "template", "data": {"template": "x"}}
    with pytest.raises(GraphError, match="Not reachable from start: orphan"):
        make_graph([START, orphan, END], [edge("start", "end"), edge("orphan", "end")])


def test_reference_must_point_at_an_earlier_node():
    a = {"id": "a", "type": "template", "data": {"template": "{{#b.output#}}"}}
    b = {"id": "b", "type": "template", "data": {"template": "b"}}
    with pytest.raises(GraphError, match="'b' doesn't run before it"):
        make_graph(
            [START, a, b, END],
            [edge("start", "a"), edge("start", "b"), edge("a", "end"), edge("b", "end")],
        )


def test_reference_to_undeclared_input_is_rejected():
    t = {"id": "t", "type": "template", "data": {"template": "{{#start.typo#}}"}}
    with pytest.raises(GraphError, match="input 'typo'"):
        make_graph([START, t, END], [edge("start", "t"), edge("t", "end")])


def test_invalid_node_settings_are_rejected():
    llm = {"id": "llm", "type": "llm", "data": {"temperature": 5}}
    graph = make_graph([START, llm, END], [edge("start", "llm"), edge("llm", "end")])
    with pytest.raises(GraphError, match="Node 'llm' has invalid settings: prompt: Field required"):
        build_nodes(graph)
