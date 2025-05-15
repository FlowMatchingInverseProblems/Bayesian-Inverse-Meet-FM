import numpy as np

def get_d_from_u(arr: np.array, points: dict):
    n = len(points)
    results = np.zeros(n)
    for i in range(n):
        x, y = points[i]
        results[i] = arr[x, y]
    return results

def create_mask_(array, rows, cols, cell_size, points_count):
    result = np.zeros((3, points_count))
    used_sectors = set()
    for k in range(points_count):
        while True:
            i = np.random.randint(rows)
            j = np.random.randint(cols)
            if (i, j) not in used_sectors:
                used_sectors.add((i, j))
                break

        start_row = i * cell_size
        end_row = (i + 1) * cell_size
        start_col = j * cell_size
        end_col = (j + 1) * cell_size

        center_row = start_row + cell_size // 2 + np.random.randint(-cell_size // 7, cell_size // 7)
        center_col = start_col + cell_size // 2 + np.random.randint(-cell_size // 7, cell_size // 7)

        result[0, k] = array[center_row, center_col]
        result[1, k] = center_row / (array.shape[0] - 1)
        result[2, k] = center_col / (array.shape[1] - 1)
    return result

def create_mask(arr, n_points):
    div = 3 + n_points // 10
    cell_size = 64 // div
    rows = div
    cols = div
    mask = create_mask_(arr, rows, cols, cell_size, n_points)
    return mask