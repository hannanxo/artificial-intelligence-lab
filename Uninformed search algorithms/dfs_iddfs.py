import re
from tabnanny import check


class BSTNode:
    def __init__(self, val=None):
        self.left = None
        self.right = None
        self.val = val

    def insert(self, val):
        if not self.val:
            self.val = val
            return

        if self.val == val:
            return

        if val < self.val:
            if self.left:
                self.left.insert(val)
                return
            self.left = BSTNode(val)
            return

        if self.right:
            self.right.insert(val)
            return
        self.right = BSTNode(val)

    def get_min(self):
        current = self
        while current.left is not None:
            current = current.left
        return current.val

    def get_max(self):
        current = self
        while current.right is not None:
            current = current.right
        return current.val

    def delete(self, val):
        if self == None:
            return self
        if val < self.val:
            if self.left:
                self.left = self.left.delete(val)
            return self
        if val > self.val:
            if self.right:
                self.right = self.right.delete(val)
            return self
        if self.right == None:
            return self.left
        if self.left == None:
            return self.right
        min_larger_node = self.right
        while min_larger_node.left:
            min_larger_node = min_larger_node.left
        self.val = min_larger_node.val
        self.right = self.right.delete(min_larger_node.val)
        return self

    def exists(self, val):
        if val == self.val:
            return True

        if val < self.val:
            if self.left == None:
                return False
            return self.left.exists(val)

        if self.right == None:
            return False
        return self.right.exists(val)

    def preorder(self, vals):
        if self.val is not None:
            vals.append(self.val)
        if self.left is not None:
            self.left.preorder(vals)
        if self.right is not None:
            self.right.preorder(vals)
        return vals

    def inorder(self, vals):
        if self.left is not None:
            self.left.inorder(vals)
        if self.val is not None:
            vals.append(self.val)
        if self.right is not None:
            self.right.inorder(vals)
        return vals

    def postorder(self, vals):
        if self.left is not None:
            self.left.postorder(vals)
        if self.right is not None:
            self.right.postorder(vals)
        if self.val is not None:
            vals.append(self.val)
        return vals

    def dfs_limited(self, vals, depth, max_depth, goal):
        if self.val:
            vals.append(self.val)

        if self.val == goal:
            print('Found goal')
            return True

        if depth + 1 <= max_depth and self.left:
            self.left.dfs_limited(vals, depth + 1, max_depth, goal)                   
        
        if depth + 1 <= max_depth and self.right:
            self.right.dfs_limited(vals, depth + 1, max_depth, goal)

        return vals

   
if __name__ == '__main__':
    nums = [12, 6, 18, 19, 21, 11, 3, 5, 4, 24, 18]
    bst = BSTNode()
    for num in nums:
        bst.insert(num)
    print("preorder:")
    print(bst.preorder([]))
    print("#")
    print("inorder:")
    print(bst.inorder([]))
    print("#")
    print("postorder:")
    print(bst.postorder([]))
    check = False
    print("Depth Limited Search:")
    print(bst.dfs_limited([], 0, 2, 18))

    graph = {
    1:[2,3,5],
    2:[4,5],
    3:[5,6],
    4:[],
    5:[1,9],
    6:[5,8],
    7:[4,5,8],
    8:[],
    9:[7,8]
    }
    path = list()

    def DFS(currentNode,destination,graph,maxDepth,curList):
        print("Checking for destination",currentNode)
        curList.append(currentNode)
        if currentNode==destination:
            return True
        if maxDepth<=0:
            path.append(curList)
            return False
        for node in graph[currentNode]:
            if DFS(node,destination,graph,maxDepth-1,curList):
                return True
            else:
                curList.pop()
        return False

    def iterativeDDFS(currentNode,destination,graph,maxDepth):
        for i in range(maxDepth):
            curList = list()
            if DFS(currentNode,destination,graph,i,curList):
                return True
        return False

    if not iterativeDDFS(1,8,graph,4):
        print("Path is not available")
    else:
        print("1 path exists")
        print(path.pop())