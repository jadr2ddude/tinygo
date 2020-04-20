package goroutine

import (
	"reflect"
	"sort"
	"testing"
)

func TestReverse(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		in, out digraph
	}{
		{
			name: "nil",
		},
		{
			name: "single",
			in: digraph{
				0: nil,
			},
			out: digraph{
				0: nil,
			},
		},
		{
			name: "selfloop",
			in: digraph{
				0: {0},
			},
			out: digraph{
				0: {0},
			},
		},
		{
			name: "simple",
			in: digraph{
				0: {1},
				1: nil,
			},
			out: digraph{
				0: nil,
				1: {0},
			},
		},
		{
			name: "cycle",
			in: digraph{
				0: {1},
				1: {0},
			},
			out: digraph{
				0: {1},
				1: {0},
			},
		},
	}

	for _, c := range cases {
		caseDat := c
		t.Run(caseDat.name, func(t *testing.T) {
			t.Parallel()
			out := caseDat.in.reverse()
			if !graphEqual(out, caseDat.out) {
				t.Errorf("got %v; want %v", out, caseDat.out)
			}
		})
	}
}

func TestTransitiveClosure(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		in, out digraph
	}{
		{
			name: "nil",
		},
		{
			name: "single",
			in: digraph{
				0: nil,
			},
			out: digraph{
				0: nil,
			},
		},
		{
			name: "selfloop",
			in: digraph{
				0: {0},
			},
			out: digraph{
				0: {0},
			},
		},
		{
			name: "simple",
			in: digraph{
				0: {1},
				1: nil,
			},
			out: digraph{
				0: {1},
				1: nil,
			},
		},
		{
			name: "cycle",
			in: digraph{
				0: {1},
				1: {0},
			},
			out: digraph{
				0: {0, 1},
				1: {0, 1},
			},
		},
		{
			name: "tree",
			in: digraph{
				0: nil,
				1: {0},
				2: {0},
				3: {1},
				4: {1},
				5: {2},
				6: {2},
			},
			out: digraph{
				0: nil,
				1: {0},
				2: {0},
				3: {0, 1},
				4: {0, 1},
				5: {0, 2},
				6: {0, 2},
			},
		},
	}

	for _, c := range cases {
		caseDat := c
		t.Run(caseDat.name, func(t *testing.T) {
			t.Parallel()
			out := caseDat.in.transitiveClosure()
			if !graphEqual(out, caseDat.out) {
				t.Errorf("got %v; want %v", out, caseDat.out)
			}
		})
	}
}

func normalizeSCC(scc [][]int) {
	for _, component := range scc {
		sort.IntSlice(component).Sort()
	}
	sort.Slice(scc, func(i, j int) bool {
		return scc[i][0] < scc[j][0]
	})
}

func copySCC(in [][]int) [][]int {
	out := make([][]int, len(in))
	for i, v := range in {
		out[i] = append([]int(nil), v...)
	}
	return out
}

func sccEqual(x, y [][]int) bool {
	x = copySCC(x)
	y = copySCC(y)
	normalizeSCC(x)
	normalizeSCC(y)
	return reflect.DeepEqual(x, y)
}

func TestStronglyConnectedComponents(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		in   digraph
		out  [][]int
	}{
		{
			name: "nil",
		},
		{
			name: "acyclic",
			in: digraph{
				0: {1, 2},
				1: {2},
				2: nil,
			},
			out: [][]int{
				{0},
				{1},
				{2},
			},
		},
		{
			name: "selfloop",
			in: digraph{
				0: {0},
			},
			out: [][]int{
				{0},
			},
		},
		{
			name: "cyclic",
			in: digraph{
				// https://en.wikipedia.org/wiki/Strongly_connected_component#/media/File:Scc.png
				0: {1},
				1: {2, 4, 5},
				2: {3, 6},
				3: {2, 7},
				4: {0, 5},
				5: {6},
				6: {5},
				7: {3, 6},
			},
			out: [][]int{
				{0, 1, 4},
				{2, 3, 7},
				{5, 6},
			},
		},
	}

	for _, c := range cases {
		caseDat := c
		t.Run(caseDat.name, func(t *testing.T) {
			t.Parallel()
			out := caseDat.in.stronglyConnectedComponents()
			if !sccEqual(out, caseDat.out) {
				t.Errorf("got %v; want %v", out, caseDat.out)
			}
		})
	}
}

func unorderedEqual(x, y []int) bool {
	x = append([]int(nil), x...)
	y = append([]int(nil), y...)
	sort.IntSlice(x).Sort()
	sort.IntSlice(y).Sort()
	return reflect.DeepEqual(x, y)
}

func TestFindLoopedVertices(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		in   digraph
		out  []int
	}{
		{
			name: "nil",
		},
		{
			name: "acyclic",
			in: digraph{
				0: {1, 2},
				1: {2},
				2: nil,
			},
		},
		{
			name: "selfloop",
			in: digraph{
				0: {0},
			},
			out: []int{0},
		},
		{
			name: "cyclic",
			in: digraph{
				// https://en.wikipedia.org/wiki/Strongly_connected_component#/media/File:Scc.png
				0: {1},
				1: {2, 4, 5},
				2: {3, 6},
				3: {2, 7},
				4: {0, 5},
				5: {6},
				6: {5},
				7: {3, 6},
			},
			out: []int{0, 1, 2, 3, 4, 5, 6, 7},
		},
	}

	for _, c := range cases {
		caseDat := c
		t.Run(caseDat.name, func(t *testing.T) {
			t.Parallel()
			out := caseDat.in.findLoopedVertices()
			if !unorderedEqual(out, caseDat.out) {
				t.Errorf("got %v; want %v", out, caseDat.out)
			}
		})
	}
}
