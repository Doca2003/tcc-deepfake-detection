from PIL import Image, ImageChops
import io

def apply_ela(image_path, output_path, quality=90):
    original = Image.open(image_path).convert("RGB")

    buffer=io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    diff = ImageChops.difference(original, recompressed)

    max_diff = max(channel_max for _, channel_max in diff.getextrema())
    scale = 255.0/max(max_diff, 1)
    ela_image = diff.point(lambda p: min(255, int(p * scale)))

    ela_image.save(output_path)