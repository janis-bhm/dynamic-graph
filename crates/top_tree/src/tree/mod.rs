use std::{
    hash::Hash,
    ops::{Index, IndexMut},
};

use indexmap::IndexMap;

pub struct Node<V> {
    /// weight associated with a vertex.
    pub weight: V,
    /// The first edge in the list of edges incident to this node.
    /// This is an index into the `edges` vector of the tree.
    next_edge: usize,
    /// The first label in the list of labels incident to this node.
    /// This is an index into the `labels` vector of the tree.
    next_label: usize,
}

impl<V> Node<V> {
    #[cfg_attr(not(test), expect(unused))]
    fn next_edge(&self) -> Option<usize> {
        if self.next_edge == NO_EDGE {
            None
        } else {
            Some(self.next_edge)
        }
    }

    #[expect(unused)]
    fn next_label(&self) -> Option<usize> {
        if self.next_label == NO_EDGE {
            None
        } else {
            Some(self.next_label)
        }
    }
}

pub struct Edge<W> {
    /// weight associated with an edge.
    pub weight: W,
    /// The two endpoints of the edge. These are indices into the `nodes` map of the tree.
    endpoints: Endpoints,
    /// next edge in the list of edges incident to the [first, second] endpoint.
    next: EdgeLinks,
}

impl<W> Edge<W> {
    #[expect(unused)]
    fn next_for(&self, endpoint: usize) -> usize {
        self.endpoints
            .direction_of(endpoint)
            .map(|dir| self.next[dir])
            .unwrap_or_else(|| {
                panic!(
                    "Node index {} not found in edge endpoints {:?}",
                    endpoint, self.endpoints
                )
            })
    }

    pub fn endpoints(&self) -> (usize, usize) {
        (
            self.endpoints[Direction::Left],
            self.endpoints[Direction::Right],
        )
    }
}

pub struct Label<W> {
    pub weight: W,
    node: usize,
    /// next label in the list of labels incident to attache node.
    next: usize,
}

impl<W> Label<W> {
    pub fn node_id(&self) -> usize {
        self.node
    }
}

/// A tree data structure representing a forest of spanning trees over some
/// graph. Each edge and node can have an associated weight, and nodes can have
/// associated labels which act like leaf edges and may represent non-tree edges
/// in the underlying graph.
pub struct Tree<N, W = (), V = ()> {
    edges: Vec<Edge<W>>,
    labels: Vec<Label<W>>,
    nodes: IndexMap<N, Node<V>>,
}

impl<N, W, V> Tree<N, W, V> {
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            labels: Vec::new(),
            nodes: IndexMap::new(),
        }
    }

    pub fn add_node(&mut self, node: N, weight: V) -> usize
    where
        N: Eq + Hash,
    {
        self.nodes
            .entry(node)
            .insert_entry(Node {
                weight,
                next_edge: NO_EDGE,
                next_label: NO_EDGE,
            })
            .index()
    }

    pub fn add_edge(&mut self, u: usize, v: usize, w: W) -> usize {
        let [(_, un), (_, vn)] = self.nodes.get_disjoint_indices_mut([u, v]).unwrap();

        let edge = self.edges.push_idx(Edge {
            weight: w,
            endpoints: Endpoints([u, v]),
            next: EdgeLinks([un.next_edge, vn.next_edge]),
        });

        un.next_edge = edge;
        vn.next_edge = edge;

        edge
    }

    pub fn node(&self, node: &N) -> Option<&Node<V>>
    where
        N: Eq + Hash,
    {
        self.nodes.get(node)
    }

    pub fn node_index(&self, node: usize) -> Option<(&N, &Node<V>)> {
        self.nodes.get_index(node)
    }

    pub fn node_index_of(&self, node: &N) -> Option<usize>
    where
        N: Eq + Hash,
    {
        self.nodes.get_index_of(node)
    }

    pub fn edge_index(&self, edge: usize) -> Option<&Edge<W>> {
        self.edges.get(edge)
    }

    pub fn edge_index_of(&self, u: usize, v: usize) -> Option<usize> {
        let (_, un) = self.nodes.get_index(u)?;

        for edge in (EdgeWalker {
            edges: &self.edges,
            current_edge: un.next_edge,
            direction: Direction::Left,
        }) {
            if edge.endpoints[Direction::Right] == v {
                return Some(edge.endpoints[Direction::Left]);
            }
        }

        None
    }

    pub fn remove_node(&mut self, node: &N) -> Option<V>
    where
        N: Eq + Hash,
    {
        let idx = self.nodes.get_index_of(node)?;

        loop {
            let next = self.nodes[idx].next_edge;
            if next == NO_EDGE {
                break;
            }

            let Some(_) = self.remove_edge(next) else {
                panic!("Edge index {} not found in tree", next);
            };
        }

        let node = self.nodes.swap_remove_index(idx).unwrap().1;

        if let Some((_, n)) = self.nodes.get_index(idx) {
            for direction in Directions {
                for edge in (EdgeWalkerMut {
                    edges: &mut self.edges,
                    current_edge: n.next_edge,
                    direction,
                }) {
                    debug_assert_eq!(
                        edge.endpoints[direction],
                        self.nodes.len(),
                        "Edge endpoint index {} does not match swapped-in node index {}",
                        edge.endpoints[direction],
                        self.nodes.len()
                    );
                    edge.endpoints[direction] = idx;
                }
            }
        }

        Some(node.weight)
    }

    pub fn add_label(&mut self, node: usize, w: W) -> usize {
        let (_, n) = self.nodes.get_index_mut(node).unwrap();

        let label = self.labels.push_idx(Label {
            weight: w,
            node,
            next: n.next_label,
        });

        n.next_label = label;

        label
    }

    pub fn remove_label(&mut self, label: usize) -> Option<W> {
        let (node, next) = {
            let l = self.labels.get(label)?;
            (l.node, l.next)
        };

        self.fix_label_links(node, label, next);
        Some(self.swap_remove_label(label))
    }

    fn fix_label_links(&mut self, node: usize, label: usize, next: usize) {
        let (_, n) = self.nodes.get_index_mut(node).unwrap();

        let first = n.next_label;

        if first == label {
            n.next_label = next;
        } else {
            for current in (LabelWalkerMut {
                labels: &mut self.labels,
                current_label: first,
            }) {
                if current.next == label {
                    current.next = next;
                    break;
                }
            }
        }
    }

    fn swap_remove_label(&mut self, idx: usize) -> W {
        let label = self.labels.swap_remove(idx);

        match self.labels.get(idx) {
            None => label.weight,
            Some(l) => {
                self.fix_label_links(l.node, self.labels.len(), l.next);
                label.weight
            }
        }
    }

    pub fn remove_edge(&mut self, edge: usize) -> Option<W> {
        let (node, next) = {
            let e = self.edges.get(edge)?;
            (e.endpoints, e.next)
        };

        self.fix_edge_links(node, edge, next);
        Some(self.swap_remove_edge(edge))
    }

    /// Removes the edge at `idx` from the `edges` vector by swapping it with
    /// the last edge and popping it off, returning the weight of the removed
    /// edge and fixing any links that referenced the swapped-in edge.
    fn swap_remove_edge(&mut self, idx: usize) -> W {
        let edge = self.edges.swap_remove(idx);

        match self.edges.get(idx) {
            None => edge.weight,
            Some(e) => {
                self.fix_edge_links(e.endpoints, self.edges.len(), EdgeLinks([idx, idx]));
                edge.weight
            }
        }
    }

    /// Fix all links referencing `edge` in the edge lists of `nodes` to point to `next` instead.
    fn fix_edge_links(&mut self, nodes: Endpoints, edge: usize, next: EdgeLinks) {
        for dir in Directions {
            let node = self
                .nodes
                .get_index_mut(nodes[dir])
                .expect("Node index not found in tree")
                .1;

            let first = node.next_edge;

            if first == edge {
                node.next_edge = next[dir];
            } else {
                for current in (EdgeWalkerMut {
                    edges: &mut self.edges,
                    current_edge: first,
                    direction: dir,
                }) {
                    if current.next[dir] == edge {
                        current.next[dir] = next[dir];
                        break;
                    }
                }
            }
        }
    }
}

impl<N, W, V> Default for Tree<N, W, V> {
    fn default() -> Self {
        Self::new()
    }
}

const NO_EDGE: usize = usize::MAX;

pub struct LabelWalkerMut<'a, W> {
    labels: &'a mut Vec<Label<W>>,
    current_label: usize,
}

impl<'a, W> Iterator for LabelWalkerMut<'a, W> {
    type Item = &'a mut Label<W>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_label == NO_EDGE {
            return None;
        }

        let label_index = self.current_label;
        let label = self
            .labels
            .get_mut(label_index)
            .expect("Label index not found in tree");
        self.current_label = label.next;

        // SAFETY: we can safely return a reference 'a because we hold the
        // vector of labels mutable for 'a, and we guarantee that we will not
        // return the same label twice in this iterator.
        let label = unsafe { &mut *(label as *mut Label<W>) };
        Some(label)
    }
}

struct EdgeWalker<'a, W> {
    edges: &'a Vec<Edge<W>>,
    current_edge: usize,
    direction: Direction,
}

impl<'a, W> EdgeWalker<'a, W> {
    #[cfg_attr(not(test), expect(unused))]
    fn from_node<N, V>(tree: &'a Tree<N, W, V>, node_index: usize) -> Self {
        let node = tree
            .node_index(node_index)
            .expect("Node index not found in tree")
            .1;
        match node.next_edge() {
            Some(edge_index) => {
                let edge = tree.edges.get(edge_index).unwrap();
                let direction = unsafe { edge.endpoints.direction_of_unchecked(node_index) };

                Self {
                    edges: &tree.edges,
                    current_edge: node.next_edge,
                    direction,
                }
            }
            None => Self {
                edges: &tree.edges,
                current_edge: NO_EDGE,
                direction: Direction::Left,
            },
        }
    }
}

impl<'a, W> Iterator for EdgeWalker<'a, W> {
    type Item = &'a Edge<W>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_edge == NO_EDGE {
            return None;
        }

        let edge_index = self.current_edge;
        let edge = self
            .edges
            .get(edge_index)
            .expect("Edge index not found in tree");
        self.current_edge = edge.next[self.direction];

        // SAFETY: we can safely return a reference 'a because we hold the
        // vector of edges mutable for 'a, and we guarantee that we will not
        // return the same edge twice in this iterator.
        let edge = unsafe { &*(edge as *const Edge<W>) };
        Some(edge)
    }
}

struct EdgeWalkerMut<'a, W> {
    edges: &'a mut Vec<Edge<W>>,
    current_edge: usize,
    direction: Direction,
}

impl<'a, W> Iterator for EdgeWalkerMut<'a, W> {
    type Item = &'a mut Edge<W>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_edge == NO_EDGE {
            return None;
        }

        let edge_index = self.current_edge;
        let edge = self
            .edges
            .get_mut(edge_index)
            .expect("Edge index not found in tree");
        self.current_edge = edge.next[self.direction];

        // SAFETY: we can safely return a reference 'a because we hold the
        // vector of edges mutable for 'a, and we guarantee that we will not
        // return the same edge twice in this iterator.
        let edge = unsafe { &mut *(edge as *mut Edge<W>) };
        Some(edge)
    }
}

struct Directions;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Direction {
    Left,
    Right,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Endpoints([usize; 2]);

impl Endpoints {
    #[expect(unused)]
    fn direction_of(&self, node: usize) -> Option<Direction> {
        if self.0[0] == node {
            Some(Direction::Left)
        } else if self.0[1] == node {
            Some(Direction::Right)
        } else {
            None
        }
    }

    #[cfg_attr(not(test), expect(unused))]
    fn contains(&self, node: usize) -> bool {
        self.0[0] == node || self.0[1] == node
    }

    #[cfg_attr(not(test), expect(unused))]
    unsafe fn direction_of_unchecked(&self, node: usize) -> Direction {
        if self.0[0] == node {
            Direction::Left
        } else {
            debug_assert_eq!(self.0[1], node, "Node index not found in endpoints");
            Direction::Right
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct EdgeLinks([usize; 2]);

impl Index<Direction> for Endpoints {
    type Output = usize;

    fn index(&self, index: Direction) -> &Self::Output {
        match index {
            Direction::Left => &self.0[0],
            Direction::Right => &self.0[1],
        }
    }
}

impl IndexMut<Direction> for Endpoints {
    fn index_mut(&mut self, index: Direction) -> &mut Self::Output {
        match index {
            Direction::Left => &mut self.0[0],
            Direction::Right => &mut self.0[1],
        }
    }
}

impl Index<Direction> for EdgeLinks {
    type Output = usize;

    fn index(&self, index: Direction) -> &Self::Output {
        match index {
            Direction::Left => &self.0[0],
            Direction::Right => &self.0[1],
        }
    }
}

impl IndexMut<Direction> for EdgeLinks {
    fn index_mut(&mut self, index: Direction) -> &mut Self::Output {
        match index {
            Direction::Left => &mut self.0[0],
            Direction::Right => &mut self.0[1],
        }
    }
}

impl IntoIterator for Directions {
    type Item = Direction;

    type IntoIter = std::array::IntoIter<Direction, 2>;

    fn into_iter(self) -> Self::IntoIter {
        [Direction::Left, Direction::Right].into_iter()
    }
}

trait VecExt {
    type Item;
    fn push_idx(&mut self, item: Self::Item) -> usize;

    #[expect(unused)]
    /// Removes the element at `index` from the vector by swapping it with the last element and popping it off, returning the value and an optional mutable reference to the new element at the swapped-out index, if it exists.
    /// ```rust
    /// let mut vec = vec![1, 2, 3];
    /// assert_eq!(vec.swap_remove_idx(1), (2, Some(&mut 3)));
    /// assert_eq!(vec.swap_remove_idx(1), (3, None));
    /// ```
    fn swap_remove_idx(&mut self, index: usize) -> (Self::Item, Option<&mut Self::Item>);
}
impl<T> VecExt for Vec<T> {
    type Item = T;

    fn push_idx(&mut self, item: Self::Item) -> usize {
        let idx = self.len();
        self.push(item);
        idx
    }

    fn swap_remove_idx(&mut self, index: usize) -> (Self::Item, Option<&mut Self::Item>) {
        let t = self.swap_remove(index);

        (t, self.get_mut(index))
    }
}

#[cfg(test)]
mod tests;
