# Data Source

## 1. Download Task Data

### 1.1 GDP Data

Grid： 1km x 1km

https://github.com/thestarlab/ChinaGDP/tree/master

### 1.2 Carbon Data

Grid：1km x 1km

https://db.cger.nies.go.jp/dataset/ODIAC/DL_odiac2023.html

### 1.3 Population Data

Grid：3 Arc (100m x 100m)

https://hub.worldpop.org/geodata/summary?id=49730

## 2. Task Data Processing

Run `ppp_data_solver` in data/taskdata to process the task data

In ppp_data_solver, the city and task should specifed by the parameter ''city'' and ''task'' respectively.

## 3. Image Data Collecting

Run `image_donwnloader` in data/image_data to collect the image data

In image_downloader, the city should specifed by the parameter ''place''.
