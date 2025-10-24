import itertools as itt


__STANDABLE = {'X', 'S', '%', 't', 'T', 'b', 'B', 'Q', '@', 'U', 'L'}

def check(left, right):
    playable = False
    for i in range(right.h):
        for j in range(right.w):
            if right[i][j] in __STANDABLE:
                x, y = i, j
                break
        pass
    pass
