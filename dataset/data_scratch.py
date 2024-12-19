import os
import requests
import json
import cv2
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from fake_useragent import UserAgent

# 初始化COCO数据结构
coco_data = {
    'images': [],
    'annotations': [],
    'categories': [
        {'id': 1, 'name': 'standing', 'supercategory': 'human'},
        {'id': 2, 'name': 'sitting', 'supercategory': 'human'},
        {'id': 3, 'name': 'falling', 'supercategory': 'human'},
        {'id': 4, 'name': 'walking', 'supercategory': 'human'}
    ],
    'info': {
        'description': 'Custom COCO Dataset for Human Actions',
        'version': '1.0',
        'year': 2024
    }
}

# 初始化图像ID和注释ID
image_id = 1
annotation_id = 1

# 搜索关键词列表
search_terms = ['person standing', 'person sitting', 'person falling', 'person walking']

# 创建文件夹保存图像
if not os.path.exists('images'):
    os.makedirs('images')

# 创建UserAgent实例
ua = UserAgent()

# 代理请求头
headers = {
    'User-Agent': ua.random
}


def get_image_urls(query, num_images=10):
    search_url = f"https://www.bing.com/images/search?q={query}"
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')
    image_urls = []

    # 取前 num_images 个图像URL
    for img_tag in img_tags[:num_images]:
        img_url = img_tag.get('src')
        if img_url:
            image_urls.append(img_url)
    return image_urls


# 下载图像并保存到本地
def download_image(img_url, img_name):
    try:
        img_data = requests.get(img_url, headers=headers).content
        img_path = f'images/{img_name}'
        with open(img_path, 'wb') as file:
            file.write(img_data)
        return img_path
    except Exception as e:
        print(f"Error downloading {img_url}: {e}")
        return None


# 获取图像的宽高
def get_image_dimensions(img_path):
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    return width, height


# 下载图像并生成COCO格式标注
for i, term in enumerate(search_terms):
    print(f"Searching for: {term}")
    image_urls = get_image_urls(term)

    for img_url in image_urls:
        # 生成图像名称
        img_name = f'{image_id}.jpg'

        # 下载图像
        img_path = download_image(img_url, img_name)
        if img_path is None:
            continue

        # 获取图像尺寸
        width, height = get_image_dimensions(img_path)

        # 创建COCO格式的图像信息
        image_info = {
            'id': image_id,
            'file_name': img_name,
            'width': width,
            'height': height
        }
        coco_data['images'].append(image_info)

        # 创建COCO格式的标注信息
        annotation = {
            'id': annotation_id,
            'image_id': image_id,
            'category_id': i + 1,  # 站立、坐下、倒地、行走的类别ID
            'bbox': [50, 50, width - 100, height - 100],  # 假设边界框
            'area': (width - 100) * (height - 100),
            'iscrowd': 0
        }
        coco_data['annotations'].append(annotation)

        # 更新图像ID和注释ID
        image_id += 1
        annotation_id += 1

# 保存COCO格式的JSON文件
with open('coco_human_actions.json', 'w') as json_file:
    json.dump(coco_data, json_file, indent=4)

print("爬取完成，COCO数据集已保存为 coco_human_actions.json")
