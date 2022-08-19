import unittest
from typing import List

class Solution():
    def matmul(A: List[List], B: List[List]) -> List[List]
        m = len(A)
        n = len(A[0])
        o = len(B[0])
        result = [[0] * m for _ in range(o)]
        for i in range(m):
            for k in range(o):
                item = 0
                for j in range(n):
                    item += A[i][j] * B[j][k] 
                result[i][k] = item
        return result



