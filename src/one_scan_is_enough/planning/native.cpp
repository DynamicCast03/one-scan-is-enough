// Compiled hot loops of reference planning: A* on the planning map and the
// dual-template waypoint selection of one_scan_is_enough.planning.path_sampling.
//
// The A* open set is a binary heap ordered by (f, push counter), so ties pop in
// push order. A cell already in the open set is not pushed again when its score
// improves, diagonal moves cannot cut corners, and entering a cell costs the
// move length plus its obstacle cost.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

struct Entry {
  double f;
  int64_t counter;
  int64_t node;
};

inline bool before(const Entry& a, const Entry& b) {
  return a.f < b.f || (a.f == b.f && a.counter < b.counter);
}

constexpr int kRow[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
constexpr int kCol[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
// Diagonal moves cost 1.414, a fixed approximation of sqrt(2).
constexpr double kStep[8] = {1.414, 1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.414};

// Returns the cells of the path from start to goal, or an empty vector when unreachable.
std::vector<int64_t> search(const uint8_t* free, const double* cost, int64_t h, int64_t w,
                            int64_t sr, int64_t sc, int64_t gr, int64_t gc) {
  const int64_t start = sr * w + sc;
  const int64_t goal = gr * w + gc;
  if (!free[start] || !free[goal]) return {};
  const int64_t n = h * w;
  std::vector<double> g(n, INFINITY);
  std::vector<int64_t> came(n, -1);
  std::vector<uint8_t> in_open(n, 0);
  std::vector<Entry> heap;
  heap.reserve(1 << 16);
  g[start] = 0.0;
  heap.push_back({0.0, 0, start});
  in_open[start] = 1;
  int64_t counter = 0;
  while (!heap.empty()) {
    const int64_t current = heap.front().node;
    const Entry last = heap.back();
    heap.pop_back();
    if (!heap.empty()) {
      size_t i = 0;
      const size_t size = heap.size();
      while (true) {
        size_t child = 2 * i + 1;
        if (child >= size) break;
        if (child + 1 < size && before(heap[child + 1], heap[child])) ++child;
        if (!before(heap[child], last)) break;
        heap[i] = heap[child];
        i = child;
      }
      heap[i] = last;
    }
    in_open[current] = 0;
    if (current == goal) {
      std::vector<int64_t> path;
      for (int64_t node = current; node >= 0; node = came[node]) {
        path.push_back(node);
        if (node == start) break;
      }
      return std::vector<int64_t>(path.rbegin(), path.rend());
    }
    const int64_t r = current / w;
    const int64_t c = current % w;
    for (int m = 0; m < 8; ++m) {
      const int64_t nr = r + kRow[m];
      const int64_t nc = c + kCol[m];
      if (nr < 0 || nr >= h || nc < 0 || nc >= w) continue;
      const int64_t neighbor = nr * w + nc;
      if (!free[neighbor]) continue;
      if (kRow[m] != 0 && kCol[m] != 0 && (!free[r * w + nc] || !free[nr * w + c])) continue;
      const double obstacle = cost[neighbor];
      if (!std::isfinite(obstacle)) continue;
      const double tentative = g[current] + kStep[m] + obstacle;
      if (!(tentative < g[neighbor])) continue;
      came[neighbor] = current;
      g[neighbor] = tentative;
      if (in_open[neighbor]) continue;
      const double dr = static_cast<double>(nr - gr);
      const double dc = static_cast<double>(nc - gc);
      const Entry entry{tentative + std::sqrt(dr * dr + dc * dc), ++counter, neighbor};
      heap.push_back(entry);
      size_t i = heap.size() - 1;
      while (i > 0) {
        const size_t parent = (i - 1) / 2;
        if (!before(entry, heap[parent])) break;
        heap[i] = heap[parent];
        i = parent;
      }
      heap[i] = entry;
      in_open[neighbor] = 1;
    }
  }
  return {};
}

py::object astar(py::array_t<uint8_t, py::array::c_style> free,
                 py::array_t<double, py::array::c_style> cost, py::tuple start, py::tuple goal) {
  if (free.ndim() != 2 || cost.ndim() != 2 || free.shape(0) != cost.shape(0) ||
      free.shape(1) != cost.shape(1)) {
    throw std::invalid_argument("free and cost must be 2D arrays of the same shape");
  }
  const int64_t h = free.shape(0);
  const int64_t w = free.shape(1);
  const int64_t sr = start[0].cast<int64_t>(), sc = start[1].cast<int64_t>();
  const int64_t gr = goal[0].cast<int64_t>(), gc = goal[1].cast<int64_t>();
  if (sr < 0 || sr >= h || sc < 0 || sc >= w || gr < 0 || gr >= h || gc < 0 || gc >= w) {
    return py::none();
  }
  std::vector<int64_t> path;
  {
    py::gil_scoped_release release;
    path = search(free.data(), cost.data(), h, w, sr, sc, gr, gc);
  }
  if (path.empty()) return py::none();
  py::array_t<int64_t> cells({static_cast<py::ssize_t>(path.size()), static_cast<py::ssize_t>(2)});
  auto out = cells.mutable_unchecked<2>();
  for (size_t k = 0; k < path.size(); ++k) {
    out(k, 0) = path[k] / w;
    out(k, 1) = path[k] % w;
  }
  return std::move(cells);
}

// Floating-point modulo whose result takes the sign of the divisor, as Python's % does.
inline double python_mod(double a, double b) {
  double r = std::fmod(a, b);
  if (r != 0.0 && ((r < 0.0) != (b < 0.0))) r += b;
  return r;
}

// Dual-template selection over dense points: from each chosen point, pick the
// last later point whose relative motion best matches a forward move or a turn.
std::vector<int64_t> select(const double* xz, const double* yaw, int64_t n, double max_lateral,
                            double max_forward, double max_theta) {
  constexpr double kPi = 3.14159265358979323846;
  std::vector<int64_t> indices{0};
  while (indices.back() < n - 1) {
    const int64_t a = indices.back();
    const double ca = std::cos(yaw[a]);
    const double sa = std::sin(yaw[a]);
    int64_t best = a + 1;
    double best_score = INFINITY;
    for (int64_t b = a + 1; b < n; ++b) {
      const double dx = xz[2 * b] - xz[2 * a];
      const double dz = xz[2 * b + 1] - xz[2 * a + 1];
      const double x = ca * dx - sa * dz;
      const double y = sa * dx + ca * dz;
      const double theta = python_mod(yaw[b] - yaw[a] + kPi, 2.0 * kPi) - kPi;
      const double rx = std::fabs(x) / max_lateral;
      const double ry = std::fabs(y) / max_forward;
      const double rt = std::fabs(theta) / max_theta;
      const double score = std::min(std::max(std::max(rx, std::fabs(ry - 1.0)), rt),
                                    std::max(std::max(rx, ry), std::fabs(rt - 1.0)));
      if (score <= best_score) {
        best_score = score;
        best = b;
      }
    }
    indices.push_back(best);
  }
  return indices;
}

py::array_t<int64_t> select_waypoints(py::array_t<double, py::array::c_style> xz,
                                      py::array_t<double, py::array::c_style> yaw,
                                      double max_lateral, double max_forward, double max_theta) {
  if (xz.ndim() != 2 || xz.shape(1) != 2 || yaw.ndim() != 1 || yaw.shape(0) != xz.shape(0)) {
    throw std::invalid_argument("xz must be (N, 2) and yaw (N,)");
  }
  std::vector<int64_t> indices;
  {
    py::gil_scoped_release release;
    indices = select(xz.data(), yaw.data(), xz.shape(0), max_lateral, max_forward, max_theta);
  }
  py::array_t<int64_t> out(static_cast<py::ssize_t>(indices.size()));
  std::copy(indices.begin(), indices.end(), out.mutable_data());
  return out;
}

}  // namespace

PYBIND11_MODULE(_native, module) {
  module.doc() = "Compiled A* and waypoint selection of one_scan_is_enough.planning.";
  module.def("astar", &astar, py::arg("free"), py::arg("cost"), py::arg("start"), py::arg("goal"),
             "Return the (K, 2) row and column cells from start to goal, or None when unreachable.");
  module.def("select_waypoints", &select_waypoints, py::arg("xz"), py::arg("yaw"),
             py::arg("max_lateral"), py::arg("max_forward"), py::arg("max_theta"),
             "Return the dense indices chosen by the dual-template selection, starting at 0.");
}
