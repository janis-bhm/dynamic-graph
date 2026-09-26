//! Bridge-finding algorithm using top trees, based on "Dynamic Bridge-Finding in O(log^2n) Amortized Time" by Holm, Rotenberg and Thorup (2017).
//!
//! The bridge queries themselves are answered by the CoverLevel structure of
//! Section 4, implemented as a [`top_tree::Summary`] ([`CoverLevel`]) with a
//! lazy [`CoverTag`]. The [`FindBridge`] wrapper exposes the tree operations
//! (`link`, `cut`, `connected`, `cover`, `uncover`, `cover_level`,
//! `min_covered_edge`) and the bridge queries (`find_bridge`, ...).
//!
//! It also exposes the two auxiliary query structures of the paper:
//!
//! * [`FindBridge::find_size`] implements `FindSize` from Section 5.
//! * [`FindBridge::add_label`], [`FindBridge::remove_label`] and
//!   [`FindBridge::find_first_label`] implement the FindFirstLabel structure
//!   of Section 6.
//!
//! # Example
//!
//! ```
//! use find_bridge::FindBridge;
//!
//! let mut graph = FindBridge::new();
//! let a = graph.add_vertex(0);
//! let b = graph.add_vertex(1);
//! let c = graph.add_vertex(2);
//! graph.link(a, b);
//! graph.link(b, c);
//!
//! // Every tree edge starts out as a bridge.
//! assert_eq!(graph.find_bridge(a), Some((a, b)));
//!
//! // Cover the whole path with a (non-tree) edge; nothing is a bridge now.
//! graph.cover(a, c, 0);
//! assert_eq!(graph.find_bridge(a), None);
//!
//! // Remove the cover again.
//! graph.uncover(a, c, 0);
//! assert!(graph.find_bridge_between(a, c).is_some());
//! ```

mod cover_level;

pub use cover_level::{CoverLevel, CoverTag, Edge, FindBridge, LabelId, LabelKey, NO_COVER};
