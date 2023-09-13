import graphviz
from .tensor import Tensor


def draw_graph(root: Tensor, format='svg', rankdir='LR') -> graphviz.Digraph:
    """
    Draws graph starting from root Variable. 

    Args:
        format (str): See formats https://graphviz.org/docs/outputs/
        rankdir (str): Direction of graph layout ['LR', 'TB']
    """
    dot = graphviz.Digraph(format=format, graph_attr={'rankdir': rankdir})

    def build(root: Tensor):
        dat_name = str(id(root))
        if root.fn is not None:
            op_name = str(id(root))+root.fn.graph_label
            dot.node(name=op_name, label=root.fn.graph_label)
            dot.edge(head_name=dat_name, tail_name=op_name)
            for child in root._prev:
                if op_name is not None:
                    dot.edge(tail_name=str(id(child)), head_name=op_name)
                build(child)
        dot.node(name=str(id(root)), label='{data=%s; shape=%s | grad=%s}' % (
            str(root.data)[:15]+'...', root.data.shape, str(root.grad)[:15]+'...'), shape='record')
    build(root)

    return dot
