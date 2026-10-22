from preprocessing.face_detection import detect_and_crop_face
import os

for label in ["real", "fake"]:
    input_base = f"data/samples/frames/{label}"
    output_base = f"data/samples/faces/{label}"

    total, success = 0, 0

    for video_folder in os.listdir(input_base):
        input_dir = os.path.join(input_base, video_folder)
        output_dir = os.path.join(output_base, video_folder)
        os.makedirs(output_dir, exist_ok=True)

        for frame_file in os.listdir(input_dir):
            frame_path = os.path.join(input_dir, frame_file)
            output_path = os.path.join(output_dir, frame_file)

            total += 1
            result = detect_and_crop_face(frame_path, output_path)
            if result:
                success += 1

    print(f"{label}: {success}/{total} rostos detectados com sucesso")