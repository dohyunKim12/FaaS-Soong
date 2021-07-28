from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='/media/yoon/storage/cocodata/dataset_coco.json',
                       image_folder='/media/yoon/storage/cocodata/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/media/yoon/storage/cocodata/',
                       max_len=50)
