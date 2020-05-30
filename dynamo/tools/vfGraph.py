import igraph as ig
import numpy as np

class ConverterMixin(object):
    """Answer by FastTurtle https://stackoverflow.com/questions/18020074/convert-a-baseclass-object-into-a-subclass-object-idiomatically"""
    @classmethod
    def convert_to_class(cls, obj):
        obj.__class__ = cls


class vfGraph(ConverterMixin, ig.Graph):
    """A class for manipulating the graph creating from the transition matrix, built from the (reconstructed) vector
    field. This is a derived class from igraph's Graph.
    """

    def __init__(self, *args, **kwds):
         super(vfGraph, self).__init__(*args, **kwds)


    def build_graph(self, adj_mat):
        """build sparse diffusion graph. The adjacency matrix needs to preserves divergence.
        """

        sources, targets = adj_mat.nonzero()
        edgelist = list(zip(sources.tolist(), targets.tolist()))
        self.__init__(edgelist, edge_attrs={'weight': adj_mat.data.tolist()}, directed=True)


    def multimaxflow(self, sources, sinks):
        """Multi-source multi-sink maximum flow. Ported from https://github.com/kazumits/ddhodge/blob/master/R/graphConstr.R
        """

        v_num, e_num = self.vcount(), self.ecount()
        usrc = v_num # super-source
        usink = usrc + 1  # super-sink
        new_edges = np.hstack([np.vstack(([usrc] * len(sources), sources)), \
                               np.vstack((sinks, [usink] * len(sinks)))]).T
        self.add_vertices(2)
        self.add_edges(new_edges)
        w_sum = sum(self.es.get_attribute_values('weight')[:e_num])
        self.es['weight'] = [i if i is not None else w_sum for i in self.es['weight']]

        mf = self.maxflow(usrc, usink, self.es.get_attribute_values('weight'))
        self.es.set_attribute_values('flow', mf.flow)
        self.vs.set_attribute_values('pass', self.strength(mode="in", weights=mf.flow))
        self.delete_vertices([usrc, usink])

    # add gradop, ...
