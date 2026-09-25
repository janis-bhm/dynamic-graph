use super::*;

#[test]
fn remove_nodes() {
    let mut tree = Tree::<i32, (), (), ()>::new();
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

    tree.remove_node(&2, |_| ());
    assert_eq!(tree.edges.len(), 0);
    assert_eq!(tree.nodes.len(), 1);
    verify_tree(&tree);
}

fn verify_tree<N, W, L, V>(tree: &Tree<N, W, L, V>) {
    for (i, _node) in tree.nodes.values().enumerate() {
        for edge in EdgeWalker::from_node(tree, i) {
            assert!(edge.endpoints.contains(i));
            assert!(tree.nodes.get_index(edge.endpoints.0[0]).is_some());
            assert!(tree.nodes.get_index(edge.endpoints.0[1]).is_some());
        }
    }
}

#[test]
fn labels_are_keyed_by_unique_id() {
    let mut tree = Tree::<i32, usize, i32, ()>::new();
    let u = tree.add_node(1, ());
    let v = tree.add_node(2, ());

    tree.add_label(u, 10, 100);
    tree.add_label(u, 11, 101);
    tree.add_label(v, 12, 102);

    assert_eq!(tree.label_count(), 3);
    assert_eq!(tree.degree(u), 2);
    assert_eq!(tree.degree(v), 1);

    let mut weights: Vec<i32> = tree.incident_label_weights(u).copied().collect();
    weights.sort_unstable();
    assert_eq!(weights, vec![100, 101]);

    assert_eq!(tree.remove_label(&11).map(|(i, _)| i), Some(101));
    assert_eq!(tree.label_count(), 2);
    assert!(tree.label(&11).is_none());
    assert_eq!(tree.degree(u), 1);
    assert_eq!(tree.label_weight(&10), Some(&100));

    // Removing another label must not disturb the ids or payloads of the rest.
    assert_eq!(tree.remove_label(&10).map(|(i, _)| i), Some(100));
    assert_eq!(tree.label_weight(&12), Some(&102));
    assert_eq!(tree.label(&12).unwrap().node_id(), v);
    assert_eq!(tree.degree(v), 1);

    assert_eq!(tree.remove_label(&12).map(|(i, _)| i), Some(102));
    assert_eq!(tree.label_count(), 0);
    assert_eq!(tree.degree(v), 0);
}

#[test]
#[should_panic(expected = "label id must be globally unique")]
fn duplicate_label_id_panics() {
    let mut tree = Tree::<i32, i32, usize, ()>::new();
    let u = tree.add_node(1, ());
    tree.add_label(u, 10, 100);
    tree.add_label(u, 10, 200);
}

#[test]
fn removing_node_remaps_its_labels() {
    let mut tree = Tree::<i32, i32, usize, ()>::new();
    let u = tree.add_node(1, ());
    let v = tree.add_node(2, ());
    tree.add_label(u, 10, 100);
    tree.add_label(v, 11, 101);

    tree.remove_node(&1, |_| ());

    // `v` was swapped into `u`'s slot, so its label must now point at the new
    // node index.
    let v_index = tree.node_index_of(&2).unwrap();
    assert_eq!(tree.label(&11).unwrap().node_id(), v_index);
    assert_eq!(tree.label_weight(&11), Some(&101));
}
