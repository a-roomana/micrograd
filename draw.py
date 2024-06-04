from graphviz import Digraph

from micrograd.engine import Value


def trace(root: Value) -> tuple[set[Value], set]:
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.children:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root: Value, _format="svg", rank_dir="LR"):
    """
    _format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rank_dir in ["LR", "TB"]
    nodes, edges = trace(root)
    dot = Digraph(
        format=_format, graph_attr={"rankdir": rank_dir}
    )  # , node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(
            name=str(id(n)),
            label="{%s | data %0.4f | grad %0.4f }" % (n.label, n.data, n.grad),
            shape="record",
        )
        if n.op:
            dot.node(name=str(id(n)) + n.op, label=n.op)
            dot.edge(str(id(n)) + n.op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    return dot
