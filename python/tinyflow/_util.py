from __future__ import absolute_import as _abs
import json
from nnvm import symbol, graph

def infer_variable_shapes(net, feed_dict):
    """Inference shape of all variables in the net.

    Parameters
    ----------
    net : tf.Symbol
       The symbolic network containing all the variables.

    feed_dict : dict
       dict of placeholder to known shape

    Returns
    -------
    Generator of (var, vname, vshape)
    Enables enumeration of variables in the net with corresponding name and shape.
    """
    g = graph.create(net)
    jgraph = json.loads(g.apply('SaveJSON').json_attr('json'))
    jnode_row_ptr = jgraph["node_row_ptr"]
    jnodes = jgraph["nodes"]
    shape = [[]] * jnode_row_ptr[-1]
    nindex = {n['name']: i for i, n in enumerate(jnodes)}

    for k, v in feed_dict.items():
        node_name = k.attr("name")
        shape[jnode_row_ptr[nindex[node_name]]] = v
    g._set_json_attr("shape", shape, "list_shape")
    g = g.apply("InferShape")
    shape = g.json_attr("shape")
    ret = {}
    for v in net.list_input_variables():
        vname = v.attr("name")
        vshape = shape[jnode_row_ptr[nindex[vname]]]
        if len(vshape) == 0:
            raise ValueError("not sufficient information in feed_dict")
        yield (v, vname, vshape)
