import rasterio
from rasterio.transform import from_bounds
import numpy as np
from rasterio.warp import transform

index = 0

file_paths = ["chn_ppp_2020_constrained.tif", "odiac2023_1km_excl_intl_2004.tif", "China_GDP_2020.img"]
value_path = ["Population", "Carbon", "Gdp"]
file_path = file_paths[index]
src = rasterio.open(file_path)

src_crs = src.crs
dst_crs = 'EPSG:4326'  # 这个是84坐标系

print(src_crs)

transform_matrix = src.transform


def pixel_2_84(coord):
    res = transform(src_crs, dst_crs, [coord[0]], [coord[1]])
    return ["%.4f, %.4f" % (res[0][0], res[1][0])]


def eight4_2_origin(coord):
    temp = transform(dst_crs, src_crs, [coord[0]], [coord[1]])
    return rasterio.transform.rowcol(transform_matrix, temp[0], temp[1])


# 读取TIF文件的数据
data = src.read(1)  # 读取第一个波段的数据

# 获取TIF文件的宽度和高度
width = src.width
height = src.height

beijing_boundary = [(116.03, 40.15), (116.79, 39.75)]

beijing_boundary = [eight4_2_origin(i) for i in beijing_boundary]

row_min = int(beijing_boundary[0][0][0])
row_max = int(beijing_boundary[1][0][0])
col_min = int(beijing_boundary[0][1][0])
col_max = int(beijing_boundary[1][1][0])

print(pixel_2_84(src.xy(row_min, col_min, offset='ul')))
print(pixel_2_84(src.xy(row_max, col_max, offset='ul')))

print(row_min, row_max, col_min, col_max)
print(width, height)
print(src.xy(row_min, col_min, offset='ul'))
print(src.xy(row_max, col_max, offset='ul'))

values = []
for row in range(row_min, row_max):
    for col in range(col_min, col_max):
        top_left_latlon = pixel_2_84(src.xy(row, col, offset='ul'))
        top_right_latlon = pixel_2_84(src.xy(row, col, offset='ur'))
        bottom_left_latlon = pixel_2_84(src.xy(row, col, offset='ll'))
        bottom_right_latlon = pixel_2_84(src.xy(row, col, offset='lr'))

        # 获取对应的GDP值
        value = data[row, col]
        values.append({
            'top_left': top_left_latlon,
            'top_right': top_right_latlon,
            'bottom_left': bottom_left_latlon,
            'bottom_right': bottom_right_latlon,
            f'{value_path[index]}': value
        })

print(len(values))

# 打印部分结果示例
for i in range(10):  # 只打印前10个
    print(f"Top Left: {values[i]['top_left']}, "
          f"Top Right: {values[i]['top_right']}, "
          f"Bottom Left: {values[i]['bottom_left']}, "
          f"Bottom Right: {values[i]['bottom_right']}, "
          f"{value_path[index]}: {values[i][f'{value_path[index]}']}")
