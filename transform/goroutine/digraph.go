package goroutine

import "sort"

// digraph is a representation of a directed graph.
// Each index in the outer slice represents a vertex.
// Each value in an inner slice is an index of a successor of the corresponding vertex.
// The successor slices must not have any duplicates.
type digraph [][]int

// graphEqual checks if digraphs are equal.
// Successor slices are treated as order-independent.
// Vertices are considered equal iff their indexes are equal.
// If the graphs have differing numbers of nodes, they are considered equal.
func graphEqual(x, y digraph) bool {
	if len(x) != len(y) {
		// Graphs have different numbers of nodes.
		return false
	}

	// Scan each vertex and check that both contain the same successors.
	for v, xs := range x {
		ys := y[v]

		// Index both slices.
		xmap := map[int]struct{}{}
		for _, s := range xs {
			xmap[s] = struct{}{}
		}
		ymap := map[int]struct{}{}
		for _, s := range ys {
			ymap[s] = struct{}{}
		}

		// Scan both maps and check if every element is in the opposite map.
		for s := range xmap {
			if _, ok := ymap[s]; !ok {
				return false
			}
		}
		for s := range ymap {
			if _, ok := xmap[s]; !ok {
				return false
			}
		}
	}

	return true
}

// reverse constructs a directed graph with all edges reversed.
// The resulting graph is normalized.
func (dg digraph) reverse() digraph {
	if dg == nil {
		return nil
	}

	reversed := make(digraph, len(dg))
	for vertex, successors := range dg {
		for _, successor := range successors {
			reversed[successor] = append(reversed[successor], vertex)
		}
	}

	return reversed
}

// transitiveClosure constructs a transitive closure of the directed graph.
// An edge exists in the resulting graph iff there existed a path from the tail to the head in the original graph.
func (dg digraph) transitiveClosure() digraph {
	closure := make(digraph, len(dg))
	for vertex, originalSuccessors := range dg {
		// Search for all reachable vertices.
		worklist := append([]int(nil), originalSuccessors...)
		found := map[int]struct{}{}
		for _, s := range originalSuccessors {
			found[s] = struct{}{}
		}
		for i := 0; i < len(worklist); i++ {
			// Select the next vertex to scan.
			v := worklist[i]

			// Scan the vertex's original successors.
			for _, w := range dg[v] {
				if _, ok := found[w]; ok {
					// Vertex w has already been found.
					continue
				}

				worklist = append(worklist, w)
				found[w] = struct{}{}
			}
		}

		// Add edges pointing to the reachable vertices.
		closure[vertex] = worklist
	}
	return closure
}

// stronglyConnectedComponents finds the strongly connected components of the directed graph.
func (dg digraph) stronglyConnectedComponents() [][]int {
	// Isolate strongly connected components using Tarjan's strongly connected components algorithm.
	// https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm

	search := sccSearch{
		graph:     dg,
		nextIndex: 1,
		index:     make([]int, len(dg)),
		lowlink:   make([]int, len(dg)),
		onStack:   make([]bool, len(dg)),
	}

	for i, index := range search.index {
		if index != 0 {
			// Vertex has already been visited.
			continue
		}

		search.strongConnect(i)
	}

	return search.components
}

type sccSearch struct {
	graph          digraph
	nextIndex      int
	index, lowlink []int
	onStack        []bool
	stack          []int
	components     [][]int
}

func (s *sccSearch) strongConnect(v int) {
	// Use the smallest unused index.
	s.index[v] = s.nextIndex
	s.lowlink[v] = s.nextIndex
	s.nextIndex++

	// Push v onto the stack.
	s.stack = append(s.stack, v)
	s.onStack[v] = true

	// Scan this vertex's successors.
	for _, w := range s.graph[v] {
		switch {
		case s.index[w] == 0:
			// This successor w has not been visited.
			s.strongConnect(w)
			if s.lowlink[v] > s.lowlink[w] {
				s.lowlink[v] = s.lowlink[w]
			}
		case s.onStack[w]:
			// This successor w is in this component.
			if s.lowlink[v] > s.index[w] {
				s.lowlink[v] = s.index[w]
			}
		}
	}

	if s.lowlink[v] == s.index[v] {
		// This vertex is the root of a strongly connected component.
		// Pop the component off of the stack.
		i := len(s.stack) - 1
		for s.stack[i] != v {
			i--
		}
		component := make([]int, len(s.stack)-i)
		copy(component, s.stack[i:])
		for _, n := range component {
			s.onStack[n] = false
		}
		s.stack = s.stack[:i]
		s.components = append(s.components, component)
	}
}

// findLoopedVertices generates a list of all vertices that are in a loop.
// A loop is defined as a path from one vertex back to itself along the graph edges.
func (dg digraph) findLoopedVertices() []int {
	var loopedVertices []int

	// Scan the strongly connected components for loops.
	for _, component := range dg.stronglyConnectedComponents() {
		if len(component) > 1 {
			// The component contains multiple vertices, and therefore must be a loop.
			loopedVertices = append(loopedVertices, component...)
		} else {
			// The component contains a single vertex.
			// This is a loop iff the vertex has an edge looping back to itself.
			v := component[0]
			for _, w := range dg[v] {
				if v == w {
					loopedVertices = append(loopedVertices, v)
					break
				}
			}
		}
	}

	return loopedVertices
}

// normalize normalizes the representation of the graph by sorting the succesor lists.
func (dg digraph) normalize() {
	for _, successors := range dg {
		sort.IntSlice(successors).Sort()
	}
}

// hasEdge searches a normalized digraph to see if there is an edge from the specified tail to the specified head.
func (dg digraph) hasEdge(tail, head int) bool {
	i := sort.SearchInts(dg[tail], head)
	return i < len(dg[tail]) && dg[tail][i] == head
}
