import rasterio
from rasterio.transform import from_bounds
import numpy as np
from rasterio.warp import transform

# 打开TIF文件
file_path = "chn_ppp_2020_constrained.tif"
# file_path = "odiac2023_1km_excl_intl_2004.tif"
# file_path = "China_GDP_2020.img"
with rasterio.open(file_path) as src:
    src_crs = src.crs
    dst_crs = 'EPSG:4326'

    # 读取TIF文件的数据
    data = src.read(1)  # 读取第一个波段的数据

    # 获取TIF文件的宽度和高度
    width = src.width
    height = src.height

    beijing_boundary = [(39.75, 116.03), (40.15, 116.79)]

    # 计算每个网格的四个顶点坐标和对应的GDP值
    gdp_values = []
    for row in range(height):
        for col in range(width):
            # 计算网格的四个顶点坐标
            top_left = src.xy(row, col, offset='ul')  # 左上角
            top_right = src.xy(row, col, offset='ur')  # 右上角
            bottom_left = src.xy(row, col, offset='ll')  # 左下角
            bottom_right = src.xy(row, col, offset='lr')  # 右下角

            top_left_latlon = transform(src_crs, dst_crs, [top_left[0]], [top_left[1]])
            top_right_latlon = transform(src_crs, dst_crs, [top_right[0]], [top_right[1]])
            bottom_left_latlon = transform(src_crs, dst_crs, [bottom_left[0]], [bottom_left[1]])
            bottom_right_latlon = transform(src_crs, dst_crs, [bottom_right[0]], [bottom_right[1]])

            # 获取对应的GDP值
            gdp = data[row, col]
            gdp_values.append({
                'top_left': top_left_latlon,
                'top_right': top_right_latlon,
                'bottom_left': bottom_left_latlon,
                'bottom_right': bottom_right_latlon,
                'gdp': gdp
            })

            if len(gdp_values) > 100:
                break

    # 打印部分结果示例
    for i in range(10):  # 只打印前10个
        print(f"Top Left: {gdp_values[i]['top_left']}, "
              f"Top Right: {gdp_values[i]['top_right']}, "
              f"Bottom Left: {gdp_values[i]['bottom_left']}, "
              f"Bottom Right: {gdp_values[i]['bottom_right']}, "
              f"GDP: {gdp_values[i]['gdp']}")
