import os


txt = []


def print_directory_tree(root_dir, max_depth=4, curr_depth=0):
    # if curr_depth >= max_depth:
    #     return

    # if os.path.isfile(root_dir):
    #     print("|   " * (curr_depth - 1) + "|-- " + os.path.basename(root_dir))
    # else:
    #     print("|   " * curr_depth + "|-- " + os.path.basename(root_dir))
    #     for item in os.listdir(root_dir):
    #         item_full_path = os.path.join(root_dir, item)
    #         if os.path.isdir(item_full_path):
    #             print_directory_tree(item_full_path, max_depth, curr_depth + 1)
    #         else:
    #             print("|   " * (curr_depth + 1) + "|-- " + item)

    if curr_depth >= max_depth:
        return

    if os.path.isfile(root_dir):
        txt.append("|   " * (curr_depth - 1) + "|-- " + os.path.basename(root_dir))
    else:
        txt.append("|   " * curr_depth + "|-- " + os.path.basename(root_dir))
        for item in os.listdir(root_dir):
            item_full_path = os.path.join(root_dir, item)
            if os.path.isdir(item_full_path):
                print_directory_tree(item_full_path, max_depth, curr_depth + 1)
            else:
                txt.append("|   " * (curr_depth + 1) + "|-- " + item)

    with open("./directory-tree.md", "w") as f:
        f.writelines(t + "\n" for t in txt)


rootdir = os.getcwd()
print_directory_tree(rootdir, max_depth=4)
