# Koala.jl

A library for studying nonlocal games and related topics written in pure Julia. While there are other libraries out there that do similar things, none of them are comprehensive enough for my liking. The following features are supported:

- Liang-Doherty-like algorithms for finding good strategies for a game in the classical and entangled settings, as well as for a one-way communication complexity problem in the quantum setting
- The classical NPA hierarchy for upper-bounding the commuting operators value of a game / communication complexity problem (along with an option to specialize to strategies employing a maximally entangled state)
- The NPA hierarchy of Russell tailored for synchronous strategies, in particular for graph coloring
- The graph parameter xi_SDP, which is a lower bound on the commuting operators value of a graph

