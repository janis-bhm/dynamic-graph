//! Bridge-finding algorithm using top trees, based on "Dynamic Bridge-Finding
//! in O(log^2n) Amortized Time" by Holm, Rotenberg and Thorup (2017).
//!
//! The bridge queries themselves are answered by the [`CoverLevel`] structure of
//! Section 4, implemented as a [`top_tree::Summary`] with a lazy [`CoverTag`].
//! [`FindBridge`] owns a dynamic forest whose edges are [`EdgeId`] handles and
//! exposes
//!
//! * the tree operations of Section 2: [`FindBridge::link`],
//!   [`FindBridge::connected`], [`FindBridge::cover_level_between`] and
//!   [`FindBridge::min_covered_edge_between`], plus [`FindBridge::remove_vertex`]
//!   for vertices that are already isolated in the forest (it panics while a
//!   vertex still has incident edges);
//! * the bridge queries, [`FindBridge::find_bridge`] and
//!   [`FindBridge::find_bridge_between`], each answering with an [`EdgeId`] or
//!   `None`;
//! * `FindSize` from Section 5 as [`FindBridge::component_size`] and
//!   [`FindBridge::two_edge_component_size`];
//! * and `FindFirstLabel` from Section 6 as internal label bookkeeping:
//!   [`FindBridge::link`] on an already-connected pair records a non-tree edge
//!   instead of a tree edge, covering the tree path between the endpoints
//!   itself, and [`FindBridge::remove_edge`] drops such a record and then
//!   uncovers the path it covered.
//!
//! There is no separate graph-level type: callers drive tree edges and non-tree
//! edges through [`FindBridge`] itself. Cover levels are bookkeeping that
//! [`FindBridge::link`] and [`FindBridge::remove_edge`] maintain on the caller's
//! behalf, not something a caller ever raises or lowers.
//!
//! # Example
//!
//! ```
//! use find_bridge::FindBridge;
//!
//! let mut graph = FindBridge::new();
//! let a = graph.add_vertex();
//! let b = graph.add_vertex();
//! let c = graph.add_vertex();
//! let ab = graph.link(a, b);
//! let bc = graph.link(b, c);
//!
//! // Every tree edge starts out as a bridge.
//! assert_eq!(graph.find_bridge(a), Some(ab));
//! assert_eq!(graph.find_bridge_between(b, c), Some(bc));
//! assert_eq!(graph.tree_edge_count(), 2);
//!
//! // Linking an already-connected pair records a non-tree edge, and `link`
//! // covers the path itself: there is no separate cover step to remember.
//! let ac = graph.link(a, c);
//! assert_eq!(graph.find_bridge(a), None);
//! assert_eq!(graph.find_bridge_between(a, c), None);
//!
//! // The cycle added no forest edge, only a non-tree record, which reports
//! // the orientation it was created with.
//! assert_eq!(graph.tree_edge_count(), 2);
//! assert_eq!(graph.endpoints(ac), Some((a, c)));
//!
//! // All three vertices now lie in one 2-edge-connected component.
//! assert_eq!(graph.component_size(a), 3);
//! assert_eq!(graph.two_edge_component_size(a), 3);
//!
//! // Dropping the non-tree record uncovers the path it covered.
//! graph.remove_edge(ac);
//! assert_eq!(graph.find_bridge(a), Some(ab));
//! let bridge = graph.find_bridge_between(a, c).unwrap();
//! assert!(bridge == ab || bridge == bc);
//!
//! // The path is uncovered again, so each vertex stands alone.
//! assert_eq!(graph.two_edge_component_size(a), 1);
//! assert_eq!(graph.endpoints(ac), None);
//! ```
//!
//! Linking an already-connected pair records a non-tree edge rather than a tree
//! edge, and covers the tree path between the endpoints at level `0` itself, so
//! the bridge queries already see the cycle by the time `link` returns:
//!
//! ```
//! use find_bridge::{FindBridge, NO_COVER};
//!
//! let mut graph = FindBridge::new();
//! let a = graph.add_vertex();
//! let b = graph.add_vertex();
//! let c = graph.add_vertex();
//! let ab = graph.link(a, b);
//! graph.link(b, c);
//!
//! // An uncovered tree path has cover level -1, and every vertex is on its own.
//! assert_eq!(graph.cover_level_between(a, c), -1);
//! assert_eq!(graph.two_edge_component_size(a), 1);
//!
//! // This `link` closes the cycle. The path comes back covered without any
//! // cover call, so no edge on it is a bridge any more.
//! graph.link(c, a);
//! assert_eq!(graph.cover_level_between(a, b), 0);
//! assert_eq!(graph.cover_level_between(b, c), 0);
//! assert_eq!(graph.find_bridge(a), None);
//! assert_eq!(graph.find_bridge_between(a, c), None);
//!
//! // Covering the path merges its vertices for `FindSize` at level 0.
//! assert_eq!(graph.two_edge_component_size(a), 3);
//!
//! // `min_covered_edge_between` names an edge attaining the path minimum
//! // whatever that minimum is; only the bridge queries filter on -1.
//! assert_eq!(graph.min_covered_edge_between(a, b), Some(ab));
//!
//! // A path that does not exist is `NO_COVER`, with no edge to name.
//! let d = graph.add_vertex();
//! assert_eq!(graph.cover_level_between(a, d), NO_COVER);
//! assert_eq!(graph.min_covered_edge_between(a, d), None);
//! ```

mod cover_level;

pub use cover_level::{CoverLevel, CoverTag, EdgeId, FindBridge, Level, NO_COVER, VertexId};
