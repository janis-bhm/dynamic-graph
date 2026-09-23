use super::*;

#[test]
fn remove_nodes() {
    let mut tree = Tree::<i32, (), ()>::new();
    let u = tree.add_node(1, ());

    let v = tree.add_node(2, ());
    let e = tree.add_edge(u, v, ());
    assert_eq!(tree.edges.len(), 1);
    assert_eq!(tree.nodes.len(), 2);

    tree.remove_edge(e);

    verify_tree(&tree);

    assert_eq!(tree.edges.len(), 0);

    tree.add_edge(u, v, ());
    assert_eq!(tree.edges.len(), 1);
    verify_tree(&tree);

    tree.remove_node(&2);
    assert_eq!(tree.edges.len(), 0);
    assert_eq!(tree.nodes.len(), 1);
    verify_tree(&tree);
}

fn verify_tree<N, W, V>(tree: &Tree<N, W, V>) {
    for (i, _node) in tree.nodes.values().enumerate() {
        for edge in EdgeWalker::from_node(tree, i) {
            assert!(edge.endpoints.contains(i));
            assert!(tree.nodes.get_index(edge.endpoints.0[0]).is_some());
            assert!(tree.nodes.get_index(edge.endpoints.0[1]).is_some());
        }
    }
}
