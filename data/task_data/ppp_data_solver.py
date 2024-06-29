import rasterio
from rasterio.transform import from_bounds
import numpy as np
from rasterio.warp import transform

city = 1
boundaries = [[(116.03, 40.15), (116.79, 39.75)], [(121.1, 31.51), (121.8, 31.1)],
              [(113.1, 23.4), (113.68, 22.94)], [(113.75, 22.84), (114.62, 22.45)]]
city_names = ["Beijing", "Shanghai", "Guangzhou", "Shenzhen"]
boundary = boundaries[city]
city_name = city_names[city]

index = 2
file_paths = ["chn_ppp_2020_constrained.tif", "odiac2023_1km_excl_intl_2004.tif", "China_GDP_2020.img"]
value_paths = ["Population", "Carbon", "Gdp"]
file_path = file_paths[index]
value_path = value_paths[index]

src = rasterio.open(file_path)

src_crs = src.crs
dst_crs = 'EPSG:4326'  # 这个是84坐标系

print(src_crs)

transform_matrix = src.transform


def pixel_2_84(coord):
    res = transform(src_crs, dst_crs, [coord[0]], [coord[1]])
    return [res[0][0], res[1][0]]


def eight4_2_origin(coord):
    temp = transform(dst_crs, src_crs, [coord[0]], [coord[1]])
    return rasterio.transform.rowcol(transform_matrix, temp[0], temp[1])


def get_coords(src, row, col):
    top_left_latlon = pixel_2_84(src.xy(row, col, offset='ul'))
    top_right_latlon = pixel_2_84(src.xy(row, col, offset='ur'))
    bottom_left_latlon = pixel_2_84(src.xy(row, col, offset='ll'))
    bottom_right_latlon = pixel_2_84(src.xy(row, col, offset='lr'))
    return [top_left_latlon, top_right_latlon, bottom_left_latlon, bottom_right_latlon]


def get_bbx(coords):
    # print(coords, coords[0][0])
    x = [i[0] for i in coords]
    y = [i[1] for i in coords]
    return [(min(x), min(y)), (max(x), max(y))]


def merge_bbx(bbx1, bbx2):
    return [(min(bbx1[0][0], bbx2[0][0]), min(bbx1[0][1], bbx2[0][1])),
            (max(bbx1[1][0], bbx2[1][0]), max(bbx1[1][1], bbx2[1][1]))]


# 读取TIF文件的数据
data = src.read(1)  # 读取第一个波段的数据

# 获取TIF文件的宽度和高度
width = src.width
height = src.height

boundary = [eight4_2_origin(i) for i in boundary]

row_min = int(boundary[0][0][0])
row_max = int(boundary[1][0][0])
col_min = int(boundary[0][1][0])
col_max = int(boundary[1][1][0])

print(pixel_2_84(src.xy(row_min, col_min, offset='ul')))
print(pixel_2_84(src.xy(row_max, col_max, offset='ul')))

print(row_min, row_max, col_min, col_max)
print(width, height)
print(src.xy(row_min, col_min, offset='ul'))
print(src.xy(row_max, col_max, offset='ul'))

step = 1 if index > 0 else 10
values = []
for row in range(row_min, row_max, step):
    for col in range(col_min, col_max, step):
        if row + step > row_max or col + step > col_max:
            break

        coords = get_coords(src, row, col)
        bbx = get_bbx(coords)
        value = data[row, col]

        if step != 1:
            coords = get_coords(src, row + step - 1, col + step - 1)
            bbx = merge_bbx(bbx, get_bbx(coords))
            coords = get_coords(src, row, col + step - 1)
            bbx = merge_bbx(bbx, get_bbx(coords))
            coords = get_coords(src, row + step - 1, col)
            bbx = merge_bbx(bbx, get_bbx(coords))
            value = 0
            for i in range(row, row + step):
                for j in range(col, col + step):
                    value += data[i, j]
            value = value / step / step

        # 获取对应的GDP值

        values.append([bbx[0][0], bbx[0][1], bbx[1][0], bbx[1][1], value])

print(len(values))
print(values[:10])

np.save(f"./{city_name}/{value_path}.npy", values)
