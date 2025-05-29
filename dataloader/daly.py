import pickle
from PIL import Image
from matplotlib import transforms
from torch.utils.data import Dataset
import torch
import cv2

class DALYFramesDataset(Dataset):
    def __init__(self, annotations_file):
        self.annotations_file = annotations_file
        with open(self.annotations_file) as f:
            self.daly = pickle.load(f)
       
    def get_all_videos(self):
        return self.daly['annot'].keys()
    
    def get_video_info(self, video_name):
        video_info = []
        for action in self.daly['annot'][video_name]['annot'].keys():
            for index, instance in enumerate(self.daly['annot'][video_name]['annot'][action]):
                video_info.append({
                    'action': action,
                    'instance': index,
                    'startTime': instance.get('beginTime', None),
                    'endTime': instance.get('endTime', None)
                })
            
        return video_info

def load_clip(self, videos_files, video_name, image_size=(224, 224)):
    # take the video from the thing on the internet

    cap = cv2.VideoCapture(video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    video_instances = self.get_video_info(video_name)
    frames = []
    for instance in video_instances:
        cap.set(cv2.CAP_PROP_POS_FRAMES, instance['instance'])
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(frame)
        frames.append(tensor)
    
    cap.release()
    
    video_tensor = torch.stack(frames, dim=0)  
    return video_tensor