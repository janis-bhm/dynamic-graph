A TopTree is a data structure storing information about a dynamic forest.
Each edge carries a weight, and vertices can have associated self-edges with weights as well.
Each cluster in a TopTree represents a connected component of the forest, and their weights are computed bottom-up from the weights of the edges and self-edges in the cluster.
K clusters are merged and the merge function is provided the exact boundary of its children.

The TopTree can be queried for the weight associated with the cluster containing a given vertex v or path v...w using the `with_exposed_path`/`with_exposed_vertex` methods, which "expose" a path or vertex so that all clusters in the component containing the vertex or path vertices have the vertex or path vertices as a boundary.

This happens via at most log(n) calls to the merge and split methods of the weight.


