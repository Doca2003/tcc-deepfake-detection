from preprocessing.ela import apply_ela
import os

for label in ["real", "fake"]:
    input_base = f"data/samples/faces/{label}"
    output_base = f"data/samples/ela/{label}"

    for video_folder in os.listdir(input_base):
        input_dir = os.path.join(input_base, video_folder)
        output_dir = os.path.join(output_base, video_folder)
        os.makedirs(output_dir, exist_ok=True)

        for face_file in os.listdir(input_dir):
            apply_ela(os.path.join(input_dir, face_file),os.path.join(output_dir, face_file))

    print(f"{label} ELA aplicado")