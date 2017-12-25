# coding:utf-8
# 从图虫网下载高清的摄影图像

from urllib import request, parse
import json
import os
import time

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36"}


def get_subject():
    global headers
    url = "https://tuchong.com/rest/tag-categories/subject?page={}&count=20"
    page = 1
    pages = 8
    result = []
    while page <= pages:
        req = request.Request(url.format(page), headers=headers)
        content = request.urlopen(req).read().decode("utf8")
        response = json.loads(content)
        if response["result"] != "SUCCESS":
            return result
        data = response["data"]
        pages = data["pages"]
        tag_list = data["tag_list"]
        for tag in tag_list:
            result.append(tag["tag_name"])
        page += 1
    return result


def get_groups(tag_name):
    global headers
    url = "https://tuchong.com/rest/tags/{}/posts?page={}&count=20&order=weekly&before_timestamp="
    result = []
    page = 1
    while len(result) < 256:
        req = request.Request(url.format(parse.quote(tag_name), page), headers=headers)
        content = request.urlopen(req).read().decode("utf8")
        response = json.loads(content)
        if response["result"] != "SUCCESS":
            return result
        post_list = response["postList"]
        for post in post_list:
            if post["type"] != "multi-photo":
                continue
            result.append(post["post_id"])
        page += 1
    return result


def get_image_url(post_id):
    global headers
    url = "https://tuchong.com/rest/posts/{}"
    result = []
    req = request.Request(url.format(post_id), headers=headers)
    content = request.urlopen(req).read().decode("utf8")
    response = json.loads(content)
    if response["result"] != "SUCCESS":
        return result
    image_list = response["images"]
    for image in image_list:
        user_id = image["user_id"]
        img_id = image["img_id"]
        result.append("https://photo.tuchong.com/{}/f/{}.jpg".format(user_id, img_id))
    return result


def download_images(image_url, save_path="image/", delay=0):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    count = len(os.listdir(save_path))
    for url in image_url:
        filename = os.path.join(save_path, "%08d.jpg")
        request.urlretrieve(url, filename=filename % count)
        count += 1
        time.sleep(delay)


if __name__ == '__main__':
    subjects = get_subject()
    for sub in subjects:
        post_ids = get_groups(sub)
        for post_id in post_ids:
            urls = get_image_url(post_id)
            download_images(urls)
