# based on https://github.com/open-mmlab/mmdetection/blob/v2.28.0/demo/image_demo.py
# Copyright (c) OpenMMLab. All rights reserved.

import cv2
import time
import tempfile
from argparse import ArgumentParser

from mmdet.apis import init_detector
from libs.api.inference import inference_one_image
from libs.utils.visualizer import visualize_lanes
from get_config_dataset_culane_clrernet import get_config_dataset_culane


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('video', help='Input video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default='result.mp4', help='Path to output video file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold'
    )
    args = parser.parse_args()
    return args


def main(args):

    # Open input video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {args.video}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize the model
    cfg_options = {
        'model.test_cfg.ori_img_w': width,
        'model.test_cfg.ori_img_h': height,
        'data': get_config_dataset_culane(width, height)
    }

    model = init_detector(args.config, args.checkpoint,
                          device=args.device, cfg_options=cfg_options)

    # Define video writer for output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out_file, fourcc, fps, (width, height))

    # Process video frame by frame
    total_inference_time = 0.0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
    
            # Save frame to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.png') as temp_img_file:
                temp_img_path = temp_img_file.name
                cv2.imwrite(temp_img_path, frame)
    
                # Profile inference time
                start_time = time.time()
                src, preds = inference_one_image(model, temp_img_path)
                end_time = time.time()
    
            inference_time = end_time - start_time
            total_inference_time += inference_time
            frame_count += 1
            avg_inference_time = total_inference_time / frame_count
    
            print(f"Average inference time after {frame_count} frames: {avg_inference_time:.4f} seconds")
    
            # Visualize the results
            print(f"Predictions: {len(preds)}")
            dst = visualize_lanes(src, preds)
    
            # Write the processed frame to the output video
            out.write(dst)

    except KeyboardInterrupt:
        print("Closing...")

    # Release video objects
    cap.release()
    out.release()


if __name__ == '__main__':
    args = parse_args()
    main(args)
