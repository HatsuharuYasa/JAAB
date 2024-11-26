class COCODataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, image_dir, classes, transform=None, categories = []):
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        self.image_dir = image_dir
        self.classes = classes
        self.transform = transform

        self.image_annotations = {img['id']: [] for img in self.coco_data['images']}
        for ann in self.coco_data['annotations']:
            if ann['category_id'] in categories:
                self.image_annotations[ann['image_id']].append(ann)
        
        #Identify the background image
        filtered_images = [
            img for img in self.coco_data['images']
            if len(self.image_annotations[img['id']]) == 0
        ]

        #Select 100% of non included categories to be removed
        num_to_remove = int(len(filtered_images) * 1)
        images_to_remove = random.sample(filtered_images, num_to_remove)
        images_to_remove_ids = {img['id'] for img in images_to_remove}

        #Remove the selected images
        self.coco_data['images'] = [
            img for img in self.coco_data['images']
            if img['id'] not in images_to_remove_ids
        ]

        #Remove the annotations as well
        self.coco_data['annotations'] = [
            ann for ann in self.coco_data['annotations']
            if ann['image_id'] in {img['id'] for img in self.coco_data['images']}
        ]

        #Rebuild the annotations
        self.image_annotations = {img['id']: [] for img in self.coco_data['images']}
        for ann in self.coco_data['annotations']:
            if ann['category_id'] in categories:
                self.image_annotations[ann['image_id']].append(ann)

        self.images = self.coco_data['images']
        self.image_ids = [img['id'] for img in self.coco_data['images']]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_data = self.images[idx]
        image_id = image_data['id']
        image_path = f"{self.image_dir}/{image_data['file_name']}"
        image = Image.open(image_path).convert("RGB")


        annotations = self.image_annotations[image_id]
        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann['category_id']))
        
        if self.transform:
            original_width, original_height = image.size
            scale_x, scale_y = 256 / original_width, 256 / original_height
            image = self.transform(image)

            scaled_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                scaled_boxes.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
            boxes = scaled_boxes
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.empty((0,4))
        labels = torch.as_tensor(labels, dtype=torch.int64) if len(labels) > 0 else torch.empty((0,), dtype=torch.int64)
        image_id = torch.tensor([image_id])

        target = {
            'boxes': boxes,
            'labels': labels, 
            'image_id': image_id
        }
        return image, target