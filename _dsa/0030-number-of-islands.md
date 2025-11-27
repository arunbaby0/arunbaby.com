---
title: "Number of Islands"
day: 30
collection: dsa
categories:
  - dsa
tags:
  - graph
  - dfs
  - bfs
  - union find
  - matrix
difficulty: Medium
related_dsa_day: 30
related_ml_day: 30
related_speech_day: 30
---

**"Counting connected components in a 2D grid."**

## 1. Problem Statement

Given an `m x n` 2D binary grid `grid` which represents a map of `'1'`s (land) and `'0'`s (water), return the number of islands.

An **island** is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

**Example 1:**
```
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
```

**Example 2:**
```
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
```

## 2. DFS Solution (Most Intuitive)

**Intuition:**
- Iterate through each cell in the grid.
- When we find a `'1'`, increment the island count and use DFS to mark all connected `'1'`s as visited.

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        islands = 0
        
        def dfs(r, c):
            # Base cases
            if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
                return
            
            # Mark as visited by changing to '0'
            grid[r][c] = '0'
            
            # Explore all 4 directions
            dfs(r + 1, c)  # Down
            dfs(r - 1, c)  # Up
            dfs(r, c + 1)  # Right
            dfs(r, c - 1)  # Left
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':
                    islands += 1
                    dfs(r, c)  # Sink the entire island
        
        return islands
```

**Time Complexity:** \\(O(M \times N)\\) where M and N are dimensions.
**Space Complexity:** \\(O(M \times N)\\) for recursion stack in worst case (if entire grid is land).

## 3. BFS Solution

**Idea:** Use a queue to explore the island level by level.

```python
from collections import deque

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        islands = 0
        
        def bfs(r, c):
            queue = deque([(r, c)])
            grid[r][c] = '0'  # Mark as visited
            
            while queue:
                row, col = queue.popleft()
                
                # Check all 4 directions
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    new_r, new_c = row + dr, col + dc
                    if 0 <= new_r < rows and 0 <= new_c < cols and grid[new_r][new_c] == '1':
                        queue.append((new_r, new_c))
                        grid[new_r][new_c] = '0'  # Mark immediately to avoid duplicates
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':
                    islands += 1
                    bfs(r, c)
        
        return islands
```

**Time Complexity:** \\(O(M \times N)\\).
**Space Complexity:** \\(O(\min(M, N))\\) for the queue (worst case: diagonal configuration).

## 4. Union-Find Solution

**Intuition:** Treat each land cell as a node. Union adjacent land cells. Count the number of disjoint sets.

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = 0
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            # Union by rank
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
            self.count -= 1

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        uf = UnionFind(rows * cols)
        
        # Count land cells
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':
                    uf.count += 1
        
        # Union adjacent land cells
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':
                    idx = r * cols + c
                    
                    # Check right neighbor
                    if c + 1 < cols and grid[r][c + 1] == '1':
                        uf.union(idx, idx + 1)
                    
                    # Check down neighbor
                    if r + 1 < rows and grid[r + 1][c] == '1':
                        uf.union(idx, idx + cols)
        
        return uf.count
```

**Time Complexity:** \\(O(M \times N \cdot \alpha(M \times N))\\) where \\(\alpha\\) is the inverse Ackermann function (nearly constant).
**Space Complexity:** \\(O(M \times N)\\) for the Union-Find structure.

## 5. Iterative DFS with Explicit Stack

**Idea:** Avoid recursion overhead by using an explicit stack.

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        islands = 0
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':
                    islands += 1
                    stack = [(r, c)]
                    
                    while stack:
                        row, col = stack.pop()
                        if 0 <= row < rows and 0 <= col < cols and grid[row][col] == '1':
                            grid[row][col] = '0'
                            stack.extend([(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)])
        
        return islands
```

**Time Complexity:** \\(O(M \times N)\\).
**Space Complexity:** \\(O(M \times N)\\) for the stack.

## Deep Dive: Why Union-Find is Powerful

Union-Find (Disjoint Set Union - DSU) is overkill for this static problem, but it shines in **dynamic scenarios**.

**Problem Variant:** Islands are added one cell at a time. After each addition, report the number of islands.

**With Union-Find:**
```python
class DynamicIslands:
    def __init__(self, m, n):
        self.grid = [['0'] * n for _ in range(m)]
        self.uf = UnionFind(m * n)
        self.rows, self.cols = m, n
    
    def add_land(self, r, c):
        if self.grid[r][c] == '1':
            return self.uf.count  # Already land
        
        self.grid[r][c] = '1'
        self.uf.count += 1
        idx = r * self.cols + c
        
        # Union with adjacent lands
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] == '1':
                self.uf.union(idx, nr * self.cols + nc)
        
        return self.uf.count
```

**Time Complexity per operation:** \\(O(\alpha(M \times N)) \approx O(1)\\).

**Use Case:** Google Maps updating land/water boundaries in real-time.

## Deep Dive: Percolation Theory

The Number of Islands problem is related to **percolation** in statistical physics.

**Percolation Question:** If each cell is land with probability \\(p\\), what is the expected number of islands?

**Phase Transition:**
- If \\(p < p_c\\) (critical threshold), small isolated clusters.
- If \\(p > p_c\\), one giant connected component emerges.

**For a 2D square lattice:** \\(p_c \approx 0.5927\\).

**Algorithm:** Monte Carlo simulation with Union-Find to compute average cluster size.

## Deep Dive: Counting Islands in 3D (Voxel Grids)

**Extension:** Given a \\(M \times N \times L\\) 3D grid, count the number of 3D islands.

**Modification:** DFS/BFS now explores 6 directions (up, down, left, right, front, back).

```python
def numIslands3D(grid):
    if not grid:
        return 0
    
    m, n, l = len(grid), len(grid[0]), len(grid[0][0])
    islands = 0
    
    def dfs(x, y, z):
        if x < 0 or x >= m or y < 0 or y >= n or z < 0 or z >= l or grid[x][y][z] != 1:
            return
        grid[x][y][z] = 0
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            dfs(x + dx, y + dy, z + dz)
    
    for x in range(m):
        for y in range(n):
            for z in range(l):
                if grid[x][y][z] == 1:
                    islands += 1
                    dfs(x, y, z)
    
    return islands
```

**Use Case:** Medical imaging (detecting tumors in MRI scans).

## Deep Dive: The "Max Area of Island" Variant

**Problem:** Return the area of the largest island.

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        max_area = 0
        
        def dfs(r, c):
            if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != 1:
                return 0
            
            grid[r][c] = 0
            area = 1
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                area += dfs(r + dr, c + dc)
            
            return area
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    max_area = max(max_area, dfs(r, c))
        
        return max_area
```

## Deep Dive: Closed Islands (All Edges Must Be Water)

**Problem:** Count islands that are NOT touching the boundary.

**Strategy:**
1. First, sink all boundary-connected land (DFS from all boundary cells).
2. Then count remaining islands.

```python
class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        
        def dfs(r, c):
            if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != 0:
                return
            grid[r][c] = 1  # Mark as visited
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                dfs(r + dr, c + dc)
        
        # Sink boundary-connected land
        for r in range(rows):
            dfs(r, 0)  # Left boundary
            dfs(r, cols - 1)  # Right boundary
        for c in range(cols):
            dfs(0, c)  # Top boundary
            dfs(rows - 1, c)  # Bottom boundary
        
        # Count remaining islands
        islands = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0:
                    islands += 1
                    dfs(r, c)
        
        return islands
```

## Deep Dive: Parallel Island Counting

For massive grids (satellite imagery), we can parallelize.

**Approach: Divide and Conquer**
1. Split the grid into \\(K\\) vertical strips.
2. Count islands in each strip in parallel.
3. Merge adjacent strips and handle boundary cases.

**Challenge:** Islands that span multiple strips.

**Solution:** Use Union-Find to merge components across boundaries.

```python
import multiprocessing

def count_islands_parallel(grid, num_workers=4):
    rows, cols = len(grid), len(grid[0])
    strip_width = cols // num_workers
    
    def count_strip(start_col, end_col):
        # Count islands in this strip
        # Return islands + boundary cells for merging
        pass
    
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(count_strip, [(i * strip_width, (i + 1) * strip_width) for i in range(num_workers)])
    
    # Merge results with Union-Find
    # ...
    return total_islands
```

**Use Case:** Processing satellite imagery (Landsat, Sentinel) to detect deforestation.

## Deep Dive: Number of Distinct Islands

**Problem:** Two islands are the same if one is a translation of the other.

**Example:**
```
Island 1:  1 1    Island 2:  1 1
           1                 1

These are the SAME shape.
```

**Solution: Normalize the Shape**
```python
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        shapes = set()
        
        def dfs(r, c, r0, c0):
            if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != 1:
                return []
            
            grid[r][c] = 0
            shape = [(r - r0, c - c0)]  # Normalize coordinates
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                shape += dfs(r + dr, c + dc, r0, c0)
            
            return shape
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    shape = tuple(sorted(dfs(r, c, r, c)))
                    shapes.add(shape)
        
        return len(shapes)
```

## Deep Dive: Number of Islands II (Online Queries)

**Problem:** Given an initially empty grid, process \\(Q\\) queries of the form `addLand(r, c)`. After each query, report the number of islands.

**Optimized Solution: Union-Find**
```python
class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        uf = UnionFind(m * n)
        grid = [[0] * n for _ in range(m)]
        result = []
        
        for r, c in positions:
            if grid[r][c] == 1:
                result.append(uf.count)
                continue
            
            grid[r][c] = 1
            uf.count += 1
            idx = r * n + c
            
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                    uf.union(idx, nr * n + nc)
            
            result.append(uf.count)
        
        return result
```

**Time Complexity:** \\(O(Q \cdot \alpha(M \times N))\\) where \\(Q\\) is the number of queries.

## Deep Dive: Flood Fill Algorithm

The island sinking logic is **Flood Fill** (used in paint programs).

**Application: Image Segmentation**
```python
def flood_fill(image, sr, sc, new_color):
    original_color = image[sr][sc]
    if original_color == new_color:
        return image
    
    def dfs(r, c):
        if r < 0 or r >= len(image) or c < 0 or c >= len(image[0]) or image[r][c] != original_color:
            return
        image[r][c] = new_color
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            dfs(r + dr, c + dc)
    
    dfs(sr, sc)
    return image
```

**Use Case:** Photoshop's "Magic Wand" tool.

## Deep Dive: Diagonally Connected Islands

**Problem:** Islands are connected diagonally as well (8-connectivity instead of 4-connectivity).

**Modification:** Add 4 more directions.
```python
for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
    dfs(r + dr, c + dc)
```

**Use Case:** Image processing (connected component labeling).

## Comparison Table

| Approach | Time | Space | Modifies Grid? | Best Use Case |
|:---|:---|:---|:---|:---|
| **DFS (Recursive)** | \\(O(MN)\\) | \\(O(MN)\\) | Yes | Small grids |
| **BFS** | \\(O(MN)\\) | \\(O(\min(M,N))\\) | Yes | Prefer level-by-level |
| **DFS (Iterative)** | \\(O(MN)\\) | \\(O(MN)\\) | Yes | Avoid recursion limits |
| **Union-Find** | \\(O(MN \cdot \alpha(MN))\\) | \\(O(MN)\\) | No | Dynamic/Online queries |

## Implementation in Other Languages

**C++:**
```cpp
class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        if (grid.empty()) return 0;
        int rows = grid.size(), cols = grid[0].size();
        int islands = 0;
        
        function<void(int, int)> dfs = [&](int r, int c) {
            if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] != '1') return;
            grid[r][c] = '0';
            dfs(r + 1, c);
            dfs(r - 1, c);
            dfs(r, c + 1);
            dfs(r, c - 1);
        };
        
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (grid[r][c] == '1') {
                    ++islands;
                    dfs(r, c);
                }
            }
        }
        
        return islands;
    }
};
```

**Java:**
```java
class Solution {
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) return 0;
        int islands = 0;
        
        for (int r = 0; r < grid.length; r++) {
            for (int c = 0; c < grid[0].length; c++) {
                if (grid[r][c] == '1') {
                    islands++;
                    dfs(grid, r, c);
                }
            }
        }
        
        return islands;
    }
    
    private void dfs(char[][] grid, int r, int c) {
        if (r < 0 || r >= grid.length || c < 0 || c >= grid[0].length || grid[r][c] != '1') {
            return;
        }
        
        grid[r][c] = '0';
        dfs(grid, r + 1, c);
        dfs(grid, r - 1, c);
        dfs(grid, r, c + 1);
        dfs(grid, r, c - 1);
    }
}
```

## Top Interview Questions

**Q1: What if we're not allowed to modify the input grid?**
*Answer:*
Use a separate `visited` set to track visited cells. `visited.add((r, c))` instead of `grid[r][c] = '0'`.

**Q2: How would you handle a grid too large to fit in memory?**
*Answer:*
Stream the grid row by row. Use a "sliding window" approach where we maintain the current row and the previous row in memory. This limits space to \\(O(N)\\) (width of grid).

**Q3: Can Union-Find handle deletions (removing land)?**
*Answer:*
Standard Union-Find doesn't support "un-union". For deletions, you'd need to rebuild the Union-Find structure or use more advanced data structures like Link-Cut trees.

**Q4: What's the expected number of islands in a random grid?**
*Answer:*
If each cell is land with probability \\(p\\), and \\(p\\) is near the percolation threshold (\\(\approx 0.59\\)), expect \\(O(\sqrt{MN})\\) islands. Below threshold: many small islands. Above: one giant component.

## Key Takeaways

1. **DFS/BFS Both Work:** DFS is simpler, BFS is level-by-level.
2. **Union-Find for Dynamic:** Excels when land is added/removed over time.
3. **Flood Fill Pattern:** Fundamental in image processing and games.
4. **Variants Everywhere:** Max area, distinct shapes, closed islands, 3D, etc.
5. **Parallelization Possible:** Divide-and-conquer for satellite imagery scale.

## Summary

| Aspect | Insight |
|:---|:---|
| **Core Idea** | Count connected components in a grid |
| **Best Approach** | DFS for simplicity, Union-Find for dynamic |
| **Key Trick** | Mark cells as visited to avoid re-processing |
| **Applications** | Image segmentation, map analysis, percolation |

---

**Originally published at:** [arunbaby.com/dsa/0030-number-of-islands](https://www.arunbaby.com/dsa/0030-number-of-islands/)

*If you found this helpful, consider sharing it with others who might benefit.*


