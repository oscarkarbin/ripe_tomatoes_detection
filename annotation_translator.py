import json
import os


def coco_to_yolo (coco_json_path, output_dir):
    #Load coco json file
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    #Create yolo dirs
    os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(output_dir, "val")
    os.makedirs(labels_dir, exist_ok=True)

    # Create classes.txt
    categories = coco_data['categories']
    category_map = {category['id']: i for i, category in enumerate(categories)}
    class_names = [category['name'] for category in categories]
    #with open(os.path.join(output_dir, "classes.txt"), 'w') as f:
        #f.write("\n".join(class_names))

    images = {image['id']: image for image in coco_data['images']}

    for annotation in coco_data['annotations']:

        #get image details
        image_id = annotation['image_id']
        image = images[image_id]
        img_width = image['width']
        img_height = image['height']
        file_name = image['file_name']

        #get bounding box
        bbox = annotation['bbox']
        x_min, y_min, width, height = bbox
        center_x = (x_min + width / 2) / img_width
        center_y = (y_min + height / 2) / img_height
        norm_width = width / img_width
        norm_height = height / img_height

        category_id = annotation['category_id']
        class_id = category_map[category_id]

        label_file = os.path.join(labels_dir, os.path.splitext(file_name)[0] + ".txt")
        with open(label_file, 'a') as f:
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")


def remap_categories(coco_json_path, output_json_path):
    # Load COCO dataset
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Update annotations
    for annotation in coco_data["annotations"]:
        old_category_name = annotation["category_id"]
        match old_category_name:
            case 1:
                old_category_name = 3
            case 2:
                old_category_name = 2
            case 3:
                old_category_name = 1
            case 4:
                old_category_name = 3
            case 5:
                old_category_name = 2
            case 6:
                old_category_name = 1
        annotation["category_id"] = old_category_name


    # Save updated dataset
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)




def main():
    remap_categories("D:/tomato_ripeness_object_detection/data/laboro_tomato/laboro_tomato/annotations/test.json", "D:/tomato_ripeness_object_detection/data/laboro_tomato/laboro_tomato/annotations/test.json")
    print("remap done")
    coco_to_yolo("D:/tomato_ripeness_object_detection/data/laboro_tomato/laboro_tomato/annotations/test.json", "D:/tomato_ripeness_object_detection/data/labels_yolo")
    print("done")
if __name__ == '__main__':
    main()





