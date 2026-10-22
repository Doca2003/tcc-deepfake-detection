from preprocessing.extract_frames import extract_frames
import os

for label in ["real", "fake"]:
    input_dir = f"data/samples/videos/{label}"
    output_base = f"data/samples/frames/{label}"

    for video_file in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_file)
        output_path = os.path.join(output_base, video_file.replace(".mp4", ""))
        extract_frames(video_path, output_path, num_frames=5)