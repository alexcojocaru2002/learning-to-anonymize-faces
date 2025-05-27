import pickle
from PIL import Image
from torch.utils.data import Dataset

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
            video_info.append({
                'action': action,
                'startTime': self.daly['annot'][video_name]['annot'][action][0].get('beginTime', None),
                'endTime': self.daly['annot'][video_name]['annot'][action][0].get('endTime', None)
            })
            
        return video_info
    
    