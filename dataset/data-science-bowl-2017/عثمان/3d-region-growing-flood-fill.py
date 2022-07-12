import numpy as np # linear algebra

# `vol` your already segmented 3d-lungs, using one of the other scripts
# `mask` you can start with all 1s, and after this operation, it'll have 0's where you need to delete
# `start_point` a tuple of ints with (z, y, x) coordinates
# `epsilon` the maximum delta of conductivity between two voxels for selection
# `HU_mid` Hounsfield unit midpoint
# `HU_range` maximim distance from `HU_mid` that will be accepted for conductivity
# `fill_with` value to set in `mask` for the appropriate location in vol that needs to be flood filled

def region_grow(vol, mask, start_point, epsilon=5, HU_mid=0, HU_range=0, fill_with=1):

    sizez = vol.shape[0] - 1
    sizex = vol.shape[1] - 1
    sizey = vol.shape[2] - 1

    items = []
    visited = []

    def enqueue(item):
        items.insert(0,item)

    def dequeue():
        s = items.pop()
        visited.append(s)
        return s

    enqueue((start_point[0], start_point[1], start_point[2]))

    while not items==[]:

        z,x,y = dequeue()
        
        voxel = vol[z,x,y]
        mask[z,x,y] = fill_with
        
        if x<sizex:
            tvoxel = vol[z,x+1,y]
            if abs(tvoxel-voxel)<epsilon  and  abs(tvoxel-HU_mid)<HU_range:  enqueue((z,x+1,y))

        if x>0:
            tvoxel = vol[z,x-1,y]
            if abs(tvoxel-voxel)<epsilon  and  abs(tvoxel-HU_mid)<HU_range:  enqueue((z,x-1,y))

        if y<sizey:
            tvoxel = vol[z,x,y+1]
            if abs(tvoxel-voxel)<epsilon  and  abs(tvoxel-HU_mid)<HU_range:  enqueue((z,x,y+1))

        if y>0:
            tvoxel = vol[z,x,y-1]
            if abs(tvoxel-voxel)<epsilon  and  abs(tvoxel-HU_mid)<HU_range:  enqueue((z,x,y-1))

        if z<sizez:
            tvoxel = vol[z+1,x,y]
            if abs(tvoxel-voxel)<epsilon  and  abs(tvoxel-HU_mid)<HU_range:  enqueue((z+1,x,y))

        if z>0:
            tvoxel = vol[z-1,x,y]
            if abs(tvoxel-voxel)<epsilon  and  abs(tvoxel-HU_mid)<HU_range:  enqueue((z-1,x,y))
            
            
            
            