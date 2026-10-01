//! Bridge-finding algorithm using top trees, based on "Dynamic Bridge-Finding in O(log^2n) Amortized Time" by Holm, Rotenberg and Thorup (2017).
//!
//! The bridge queries themselves are answered by the CoverLevel structure of
//! Section 4, implemented as a [`top_tree::Summary`] ([`CoverLevel`]) with a
//! lazy [`CoverTag`]. The [`FindBridge`] wrapper exposes the tree operations
//! (`link`, `cut`, `connected`, `cover`, `uncover`, `cover_level`,
//! `min_covered_edge`, and isolated-in-the-forest vertex removal) and the
//! bridge queries (`find_bridge`, ...).
//!
//! It also exposes the two auxiliary query structures of the paper:
//!
//! * [`FindBridge::find_size`] implements `FindSize` from Section 5.
//! * [`FindBridge::add_label`], [`FindBridge::remove_label`] and
//!   [`FindBridge::find_first_label`] implement the FindFirstLabel structure
//!   of Section 6.
//! * [`DynamicGraph`] implements Appendix A's graph-level reduction on top of
//!   these tree operations. Its `remove_vertex` deletes all incident graph
//!   edges before removing the vertex; removed public handles are not reused.
//!
//! # Example
//!
//! ```
//! use find_bridge::{FindBridge, Level};
//!
//! let mut graph = FindBridge::new();
//! let a = graph.add_vertex();
//! let b = graph.add_vertex();
//! let c = graph.add_vertex();
//! graph.link(a, b);
//! graph.link(b, c);
//! let level = Level::new(0).unwrap();
//!
//! // Every tree edge starts out as a bridge.
//! assert_eq!(graph.find_bridge(a), Some((a, b)));
//!
//! // Cover the whole path with a (non-tree) edge; nothing is a bridge now.
//! graph.cover(a, c, level);
//! assert_eq!(graph.find_bridge(a), None);
//!
//! // Remove the cover again.
//! graph.uncover(a, c, level);
//! assert!(graph.find_bridge_between(a, c).is_some());
//! ```
//!
//! The higher-level [`DynamicGraph`] API manages tree and non-tree edges for
//! you:
//!
//! ```
//! use find_bridge::DynamicGraph;
//!
//! let mut graph = DynamicGraph::new();
//! let a = graph.add_vertex();
//! let b = graph.add_vertex();
//! let c = graph.add_vertex();
//! graph.insert_edge(a, b).unwrap();
//! graph.insert_edge(b, c).unwrap();
//! assert!(graph.find_bridge_between(a, c).is_some());
//!
//! // The third edge closes a cycle and covers the tree path.
//! let cycle_edge = graph.insert_edge(c, a).unwrap();
//! assert_eq!(graph.find_bridge(a), None);
//! assert!(graph.two_edge_connected(a, c));
//!
//! graph.delete_edge(cycle_edge);
//! assert!(graph.find_bridge_between(a, c).is_some());
//! ```

mod cover_level;

pub use cover_level::{
    CoverLevel, CoverTag, Edge, FindBridge, Level, NO_COVER, UserLabel,
};
