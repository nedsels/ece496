# Test Cases

## a_star
10 nodes. 36 edges. Each node has coordinate to use for h score calculation.

Test path 0 -> 9 should be cost 22. Unique path 0 -> 1 -> 2 -> 5 -> 9.

Path 8 -> 9 should be 22 (8 -> 3 -> 4 -> 7 -> 9 OR 8 -> 3 -> 4 -> 5 -> 9). 4-> 7|5 -> 9 have cost of 11.
For path reconstruction, this is non-deterministic.

Path self -> self should be 0.

 8 ----- 3 ----- 6
 |  \  /   \   /   \
 |   1       4 ---- 7
 |  / \    /  \   /  \
 0 ---- 2 ----- 5 --- 9 

Adjacency List:
0:  1(5), 2(12), 8(10)
1:  0(5), 2(5), 3(5), 8(10)
2:  0(12), 1(5), 4(5), 5(6)
3:  1(5), 4(5), 6(6), 8(6)
4:  2(5), 3(5), 5(5), 6(5), 7(6)
5:  2(6), 6(5), 7(5), 9(6)
6:  3(6), 4(5), 7(5)
7:  4(6), 5(5), 6(5), 9(5)
8:  0(10), 1(10), 3(6)
9:  5(6), 7(5)

Coordinates:
0: (0, 0)    1: (3, 4)    2: (6, 0)    3: (6, 8)    4: (9, 4)
5: (12, 0)   6: (12, 8)   7: (15, 4)   8: (0, 8)   9: (18, 0)


