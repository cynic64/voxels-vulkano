max_dist = 9
offsets = [(0, 0, 0)]

for x in range(-max_dist, max_dist + 1):
    for y in range(-max_dist, max_dist + 1):
        for z in range(-max_dist, max_dist + 1):
            if (x, y, z) not in offsets:
                offsets.append((x, y, z))

print(offsets);
