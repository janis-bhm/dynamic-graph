//! Bridge-finding algorithm using top trees, based on "Dynamic Bridge-Finding
//! in O(log^2n) Amortized Time" by Holm, Rotenberg and Thorup (2017).
//!
//! The bridge queries themselves are answered by the [`CoverLevel`] structure of
//! Section 4, implemented as a [`top_tree::Summary`] with a lazy [`CoverTag`].
//! [`FindBridge`] owns a dynamic forest whose edges are [`EdgeId`] handles and
//! exposes
//!
//! * the tree operations of Section 2: [`FindBridge::link`],
//!   [`FindBridge::cut_edge`], [`FindBridge::connected`], [`FindBridge::cover`],
//!   [`FindBridge::uncover`], [`FindBridge::cover_level_between`] and
//!   [`FindBridge::min_covered_edge_between`], plus
//!   [`FindBridge::remove_vertex`] for vertices that are already isolated in the
//!   forest (it panics while a vertex still has incident edges);
//! * the bridge queries, [`FindBridge::find_bridge`] and
//!   [`FindBridge::find_bridge_between`], each answering with an [`EdgeId`] or
//!   `None`;
//! * `FindSize` from Section 5 as [`FindBridge::component_size`] and
//!   [`FindBridge::two_edge_component_size`];
//! * and `FindFirstLabel` from Section 6 as internal label bookkeeping:
//!   [`FindBridge::link`] on an already-connected pair records a non-tree edge
//!   instead of a tree edge, [`FindBridge::cut_edge`] drops such a record, and
//!   [`FindBridge::remove_edge`] drops it and then uncovers the path it covered.
//!
//! There is no separate graph-level type: callers drive tree edges and non-tree
//! edges through [`FindBridge`] itself.
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
//! let ab = graph.link(a, b);
//! let bc = graph.link(b, c);
//! let level = Level::new(0).unwrap();
//!
//! // Every tree edge starts out as a bridge.
//! assert_eq!(graph.find_bridge(a), Some(ab));
//! assert_eq!(graph.find_bridge_between(b, c), Some(bc));
//!
//! // Cover the whole path; no edge on it is a bridge any more.
//! graph.cover(a, c, level);
//! assert_eq!(graph.find_bridge(a), None);
//! assert_eq!(graph.find_bridge_between(a, c), None);
//!
//! // Uncover again and each path edge is a bridge once more.
//! graph.uncover(a, c, level);
//! let bridge = graph.find_bridge_between(a, c).unwrap();
//! assert!(bridge == ab || bridge == bc);
//! ```
//!
//! Linking an already-connected pair records a non-tree edge rather than a tree
//! edge; the caller raises the cover level on the tree path to make the bridge
//! queries see the cycle:
//!
//! ```
//! use find_bridge::{FindBridge, Level};
//!
//! let mut graph = FindBridge::new();
//! let a = graph.add_vertex();
//! let b = graph.add_vertex();
//! let c = graph.add_vertex();
//! let ab = graph.link(a, b);
//! graph.link(b, c);
//! assert_eq!(graph.find_bridge_between(a, c), Some(ab));
//!
//! let level = Level::new(0).unwrap();
//! let cycle_edge = graph.link(c, a);
//! graph.cover(c, a, level);
//! assert_eq!(graph.find_bridge(a), None);
//! assert_eq!(graph.component_size(a), 3);
//! assert_eq!(graph.two_edge_component_size(a), 3);
//!
//! graph.uncover(c, a, level);
//! assert_eq!(graph.find_bridge(a), Some(ab));
//! assert_eq!(graph.two_edge_component_size(a), 1);
//!
//! assert!(graph.cut_edge(cycle_edge));
//! ```

mod cover_level;

pub use cover_level::{CoverLevel, CoverTag, EdgeId, FindBridge, Level, NO_COVER, VertexId};
