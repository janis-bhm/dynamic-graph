use super::*;

#[test]
fn remove_nodes() {
    let mut tree = Tree::<i32, (), ()>::new();
    let u = tree.add_node(1);

    let v = tree.add_node(2);
    let e = tree.add_edge(u, v, ());
    assert_eq!(tree.edges.len(), 1);
    assert_eq!(tree.nodes.len(), 2);

    tree.remove_edge(e);

    verify_tree(&tree);

    assert_eq!(tree.edges.len(), 0);

    tree.add_edge(u, v, ());
    assert_eq!(tree.edges.len(), 1);
    verify_tree(&tree);

    tree.remove_node(v, |_, _| (), |_, _| ());
    assert_eq!(tree.edges.len(), 0);
    assert_eq!(tree.nodes.len(), 1);
    verify_tree(&tree);
}

fn verify_tree<V, E, L>(tree: &Tree<V, E, L>) {
    for (i, _node) in tree.nodes.iter().enumerate() {
        for edge in EdgeWalker::from_node(tree, i) {
            assert!(edge.endpoints.contains(i));
            assert!(tree.nodes.get(edge.endpoints.0[0]).is_some());
            assert!(tree.nodes.get(edge.endpoints.0[1]).is_some());
        }
    }
}

#[test]
fn labels_are_keyed_by_unique_id() {
    let mut tree = Tree::<i32, i32, i32>::new();
    let u = tree.add_node(1);
    let v = tree.add_node(2);

    let l1 = tree.add_label(u, 100);
    let l2 = tree.add_label(u, 101);
    let l3 = tree.add_label(v, 102);

    assert_eq!(tree.label_count(), 3);
    assert_eq!(tree.degree(u), 2);
    assert_eq!(tree.degree(v), 1);

    let mut weights: Vec<i32> = tree.incident_label_weights(u).copied().collect();
    weights.sort_unstable();
    assert_eq!(weights, vec![100, 101]);

    assert_eq!(tree.remove_label(l2).map(|(i, _)| i), Some(101));
    assert_eq!(tree.label_count(), 2);
    assert!(tree.label(l2).is_none());
    assert_eq!(tree.degree(u), 1);
    assert_eq!(tree.label_weight(l1), Some(&100));

    // Removing another label must not disturb the ids or payloads of the rest.
    assert_eq!(tree.remove_label(l1).map(|(i, _)| i), Some(100));
    assert_eq!(tree.label_weight(l3), Some(&102));
    assert_eq!(tree.label(l3).unwrap().node_index(), v.index());
    assert_eq!(tree.degree(v), 1);

    assert_eq!(tree.remove_label(l3).map(|(i, _)| i), Some(102));
    assert_eq!(tree.label_count(), 0);
    assert_eq!(tree.degree(v), 0);
}

#[test]
fn removing_node_remaps_its_labels() {
    let mut tree = Tree::<i32, i32, i32>::new();
    let u = tree.add_node(1);
    let mut v = tree.add_node(2);
    let l0 = tree.add_label(u, 100);
    let l1 = tree.add_label(v, 101);

    if let Some((_, swap)) = tree.remove_node(
        u,
        |_, _| (),
        |_, swap| {
            for l in &mut [l0, l1] {
                l.swap(swap);
            }
        },
    ) {
        v.swap(swap);
    }

    assert_eq!(tree.label(l1).unwrap().node_index(), v.index());
    assert_eq!(tree.label_weight(l1), Some(&101));
}
