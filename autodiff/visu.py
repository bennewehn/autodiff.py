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
        if root.op is not None:
            op_name = str(id(root))+root.op.graph_label
            dot.node(name=op_name, label=root.op.graph_label)
            dot.edge(head_name=dat_name, tail_name=op_name)
            for child in root._children:
                if op_name is not None:
                    dot.edge(tail_name=str(id(child)), head_name=op_name)
                build(child)

        str_data = str(root.data)[:15]
        if root.data.size > 15:
            str_data += '...'

        str_grad = str(root.grad)[:15]
        if root.grad is not None and root.grad.size > 15:
            str_grad += '...'

        dot.node(name=str(id(root)), label='{shape=%s | data=%s | grad=%s}' % (
            root.data.shape,
            str_data,
            str_grad), shape='record')

    build(root)

    return dot
