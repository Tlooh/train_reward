import os
import json
from sklearn.model_selection import train_test_split

# 计数
def count_json_items(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return len(data)


def split_json(input_file, output_simple_file, output_complex_file):
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    simple_data = []
    complex_data = []

    for item in data:
        simple_item = {"id": item["id"], "simple": item["simple"]}
        complex_item = {"id": item["id"], "complex": item["complex"]}

        # 添加到对应的列表中
        simple_data.append(simple_item)
        complex_data.append(complex_item)

    # 写入两个新的JSON文件
    with open(output_simple_file, 'w', encoding='utf-8') as simple_file:
        json.dump(simple_data, simple_file, ensure_ascii=False, indent=4)

    with open(output_complex_file, 'w', encoding='utf-8') as complex_file:
        json.dump(complex_data, complex_file, ensure_ascii=False, indent=4)


def add_img_path(json_path, img_dir, output_json=None):
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    options = ['simple', 'complex']
    s2c_img_dir = [os.path.join(img_dir, mode) for mode in options]   

    for item in data:
        id = item['id']
        item["simple_img"] = f"{s2c_img_dir[0]}/{id}.png"
        item["complex_img"] = f"{s2c_img_dir[1]}/{id}.png"
    
    cnt = len(data)
    if output_json is None:
        output_json = os.path.join(os.path.dirname(json_path),f"S2C_{cnt}.json")

    with open(output_json, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    
    print(f"output json is saved in {output_json} successfully!")


def rearrange_ids(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for i, item in enumerate(data):
        item['id'] = i
    
    cnt = len(data)
    output_json = os.path.join(os.path.dirname(json_path),f"S2C_{cnt}.json")
    with open(output_json, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"output json is saved in {output_json} successfully!")


def rearrange_imgs(img_dir):
    img_files = os.listdir(img_dir)
    img_files = sorted(img_files, key= lambda x: int(x.split('.')[0]))
    for i, img_name in enumerate(img_files):
        new_img_name = f"{i}.png"
        new_img_path = os.path.join(img_dir, new_img_name)
        original_img_path = os.path.join(img_dir, img_name)

        os.rename(original_img_path, new_img_path)
    print("Finished!")
    

def split_train_val_test(json_file, ratios = [0.8, 0.1, 0.1], random_state=42):
    # 加载 JSON 数据
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 划分数据集
    train_data, temp_data = train_test_split(data, test_size=ratios[1] + ratios[2], random_state=random_state)
    val_data, test_data = train_test_split(temp_data, test_size=ratios[2] / (ratios[1] + ratios[2]), random_state=random_state)

    
    # 保存训练集、验证集、测试集为 JSON 文件
    save_json_dir = os.path.dirname(json_file)

    train_file_path = os.path.join(save_json_dir, f'train_{len(train_data)}.json')
    val_file_path = os.path.join(save_json_dir, f'val_{len(val_data)}.json')
    test_file_path = os.path.join(save_json_dir, f'test_{len(test_data)}.json')

    with open(train_file_path, 'w', encoding='utf-8') as train_file:
        json.dump(train_data, train_file, ensure_ascii=False, indent=4)

    with open(val_file_path, 'w', encoding='utf-8') as val_file:
        json.dump(val_data, val_file, ensure_ascii=False, indent=4)

    with open(test_file_path, 'w', encoding='utf-8') as test_file:
        json.dump(test_data, test_file, ensure_ascii=False, indent=4)

    # 打印信息
    print(f"Train data saved to: {train_file_path}")
    print(f"Validation data saved to: {val_file_path}")
    print(f"Test data saved to: {test_file_path}")


def select_json(json_path, range = [0, 10000]):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    new_data = []
    for item in data:
        id = item['id']
        if id >= range[0] and id <= range[1]:
            new_data.append(item)


    cnt = len(new_data)
    output_json = os.path.join(os.path.dirname(json_path),f"S2C_{cnt}.json")

    with open(output_json, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)

    print(f"output json is saved in {output_json} successfully!")



if __name__ == "__main__":
# 示例调用
    # rearrange_ids('../json/processed_all_114959.json')
    split_json('../json/S2C_114959.json', '../json/simple.json', '../json/complex.json')
    # add_img_path("/media/sdb/liutao/train_reward/json/S2C_114959.json","/media/sdb/liutao/datasets/s2c_images")
    split_train_val_test('../json/S2C_114959.json', ratios = [0.8, 0.1, 0.1], random_state=42)
    # select_json('../json/S2C_114959.json', range = [0, 10000])
    # print(count_json_items('../json/S2C_9578.json'))

    # split_json("../example/S2C_10001.json",'../example/simple.json', '../example/complex.json')
    # rearrange_imgs("/media/sdb/liutao/datasets/s2c_images/complex")


