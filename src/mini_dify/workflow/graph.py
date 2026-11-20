from collections import deque
from typing import Any, Literal

from pydantic import BaseModel, Field

from mini_dify.workflow.variables import find_references

NodeType = Literal[
    "start", "llm", "knowledge_retrieval", "if_else", "template", "http_request", "end"
]


class NodeConfig(BaseModel):
    """One box in the workflow: what it is, and its settings in `data`."""

    id: str = Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]{0,49}$")
    type: NodeType
    title: str = ""
    data: dict[str, Any] = {}


class Edge(BaseModel):
    """An arrow between two nodes. `source_handle` says which exit of the source it leaves
    from: "source" for normal nodes, "true" or "false" for an if/else node."""

    source: str
    target: str
    source_handle: str = "source"


class GraphSpec(BaseModel):
    nodes: list[NodeConfig] = Field(min_length=2)
    edges: list[Edge]


class GraphError(ValueError):
    """The workflow's shape is wrong: it can't be run."""


class Graph:
    """A checked workflow: nodes, edges, and a safe order to run them in."""

    def __init__(self, spec: GraphSpec):
        self.nodes: dict[str, NodeConfig] = {}
        for node in spec.nodes:
            if node.id in self.nodes:
                raise GraphError(f"Two nodes have the id {node.id!r}")
            self.nodes[node.id] = node
        self.edges = spec.edges

        starts = [n.id for n in spec.nodes if n.type == "start"]
        if len(starts) != 1:
            raise GraphError(f"A workflow needs exactly one start node, found {len(starts)}")
        self.start_id = starts[0]
        if not any(n.type == "end" for n in spec.nodes):
            raise GraphError("A workflow needs at least one end node")

        self._check_edges()
        self.order = self._topological_order()
        self._check_reachable()
        self._check_references()

    def incoming(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.edges if edge.target == node_id]

    def outgoing(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.edges if edge.source == node_id]

    def ancestors(self, node_id: str) -> set[str]:
        """Every node that always runs (or is skipped) before this one."""
        seen: set[str] = set()
        queue = deque([node_id])
        while queue:
            for edge in self.incoming(queue.popleft()):
                if edge.source not in seen:
                    seen.add(edge.source)
                    queue.append(edge.source)
        return seen

    def _check_edges(self) -> None:
        seen = set()
        for edge in self.edges:
            for end in (edge.source, edge.target):
                if end not in self.nodes:
                    raise GraphError(f"An edge points at an unknown node {end!r}")
            if edge.source == edge.target:
                raise GraphError(f"Node {edge.source!r} has an edge to itself")
            if self.nodes[edge.target].type == "start":
                raise GraphError("Nothing can lead into the start node")
            if self.nodes[edge.source].type == "end":
                raise GraphError(f"End node {edge.source!r} can't have outgoing edges")
            allowed = {"true", "false"} if self.nodes[edge.source].type == "if_else" else {"source"}
            if edge.source_handle not in allowed:
                raise GraphError(
                    f"Edge from {edge.source!r} uses handle {edge.source_handle!r}; "
                    f"allowed: {', '.join(sorted(allowed))}"
                )
            key = (edge.source, edge.source_handle, edge.target)
            if key in seen:
                raise GraphError(f"Duplicate edge {edge.source!r} -> {edge.target!r}")
            seen.add(key)

        for node in self.nodes.values():
            if node.type == "if_else":
                handles = {edge.source_handle for edge in self.outgoing(node.id)}
                if handles != {"true", "false"}:
                    raise GraphError(f"If/else node {node.id!r} needs a 'true' and a 'false' edge")
            elif node.type != "end" and not self.outgoing(node.id):
                raise GraphError(f"Node {node.id!r} leads nowhere: connect it or remove it")

    def _topological_order(self) -> list[str]:
        """Kahn's algorithm: repeatedly take a node with no unprocessed inputs.
        If some nodes are never taken, they're part of a cycle."""
        waiting = {node_id: len(self.incoming(node_id)) for node_id in self.nodes}
        queue = deque(node_id for node_id, count in waiting.items() if count == 0)
        order: list[str] = []
        while queue:
            node_id = queue.popleft()
            order.append(node_id)
            for edge in self.outgoing(node_id):
                waiting[edge.target] -= 1
                if waiting[edge.target] == 0:
                    queue.append(edge.target)
        if len(order) != len(self.nodes):
            stuck = set(self.nodes) - set(order)
            # Nodes after a cycle are stuck too. Peel those off: a node that leads to
            # no other stuck node can't be on the cycle itself.
            peeled = True
            while peeled:
                peeled = False
                for node_id in sorted(stuck):
                    if not any(edge.target in stuck for edge in self.outgoing(node_id)):
                        stuck.discard(node_id)
                        peeled = True
            raise GraphError(f"The workflow has a cycle through: {', '.join(sorted(stuck))}")
        return order

    def _check_reachable(self) -> None:
        reached = {self.start_id}
        for node_id in self.order:  # in order, so every source is decided before its targets
            if node_id in reached:
                reached.update(edge.target for edge in self.outgoing(node_id))
        unreachable = sorted(set(self.nodes) - reached)
        if unreachable:
            raise GraphError(f"Not reachable from start: {', '.join(unreachable)}")

    def _check_references(self) -> None:
        """Every {{#node.var#}} must point at a node that runs before the one using it."""
        start_inputs = {
            variable.get("name")
            for variable in self.nodes[self.start_id].data.get("variables", [])
            if isinstance(variable, dict)
        }
        for node in self.nodes.values():
            before = self.ancestors(node.id)
            for text in strings_in(node.data):
                for ref_node, name in find_references(text):
                    if ref_node not in self.nodes:
                        raise GraphError(f"Node {node.id!r} refers to an unknown node {ref_node!r}")
                    if ref_node not in before:
                        raise GraphError(
                            f"Node {node.id!r} uses {{{{#{ref_node}.{name}#}}}}, "
                            f"but {ref_node!r} doesn't run before it"
                        )
                    if ref_node == self.start_id and name not in start_inputs:
                        raise GraphError(
                            f"Node {node.id!r} uses the input {name!r}, "
                            "which the start node doesn't declare"
                        )


def strings_in(value: Any) -> list[str]:
    """Every string inside a node's settings, however deeply nested."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in strings_in(item)]
    if isinstance(value, list):
        return [text for item in value for text in strings_in(item)]
    return []
