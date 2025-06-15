from PIL import Image
import os
import sys


def create_gif_from_images(image_folder, output_gif, duration=200):
  """
  Create a GIF from a series of images in a folder.

  :param image_folder: Path to the folder containing images.
  :param output_gif: Path to save the output GIF.
  :param duration: Duration between frames in milliseconds.
  """
  # Get list of PNG images
  png_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

  # Load images
  images = [Image.open(os.path.join(image_folder, f)) for f in png_files]

  # Save as GIF
  images[0].save(
    output_gif,
    save_all=True,
    append_images=images[1:],
    duration=duration,
    loop=0  # loop forever
  )

def main():
  if len(sys.argv) != 4:
    print("Usage: python create_gif.py <image_folder> <output_gif> <duration>")
    sys.exit(1)

  image_folder = sys.argv[1]
  output_gif = sys.argv[2]
  duration = int(sys.argv[3])
  create_gif_from_images(image_folder, output_gif, duration)


if __name__ == "__main__":
  main()