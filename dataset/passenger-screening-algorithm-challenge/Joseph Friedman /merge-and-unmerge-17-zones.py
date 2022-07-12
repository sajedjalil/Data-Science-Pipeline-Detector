

'''
accepts list of lists

[
    ['00ff_Zone1', 0],
    ['00ff_Zone2', 1],
    ['00fa_Zone1', 1],
    ['00fa_Zone2', 0]


]

returns list of lists

[
    ['00ff',[0,1]],
    ['00fa',[1,0]]
]

'''
def merge_17(list):
    unmerged = {row[0]:row[1] for row in list}
    merged = []
    ids = id_list(list)
    for i in ids:
        tmp = []
        for j in range(1,18):
            key = i+'_Zone'+str(j)
            tmp.append(unmerged[key])
        merged.append([i,tmp])
    return merged
'''
 reverse of merge_17() above and sort
'''
def unmerge_17(merged):
    unmerged = []
    for row in merged:
        for j in range(17):
           key = row[0]+'_Zone'+str(j+1)
           unmerged.append([key,row[1][j]])
    srt = sorted(unmerged,key=lambda x: x[0])
    return srt

'''
returns list of image ids without duplicates
'''
def id_list(list):
    ids = []

    for row in list[1:]:

        tmp = row[0].split('_')

        if tmp[0] not in ids:
            ids.append(tmp[0])

    return ids
