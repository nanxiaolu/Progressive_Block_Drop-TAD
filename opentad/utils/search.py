def search_drop_idx(state_dict, depth) -> tuple:
    have_dropped_idx = []
    next_drop_idx_list = []
    temp_flag = [False for _ in range(depth)]
    for key, _ in state_dict.items():
        if 'backbone' in key:
            for i in range(depth):
                if f'backbone.blocks.{i}.attn' in key:
                    temp_flag[i] = True
                    break

    for i in range(depth):
        if not temp_flag[i]:
            have_dropped_idx.append(i)

    for i in range(depth):
        if i not in have_dropped_idx:
            temp = []
            temp.append(i)
            next_drop_idx_list.append(temp.copy())

    print(f"drop list is: {next_drop_idx_list}")
    return next_drop_idx_list
