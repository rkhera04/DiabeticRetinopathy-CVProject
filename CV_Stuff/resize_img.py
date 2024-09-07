from PIL import Image
import os

def resize_image(input_path, output_path, size=(650, 650)):

    try:
        with Image.open(input_path) as img:
            img = img.resize(size, Image.ANTIALIAS)
            img.save(output_path)
            print(f"Image successfully resized and saved to {output_path}")
    except Exception as e:
        print(f"An error occurred with {input_path}: {e}")

def resize_images_in_folder(input_folder, output_folder, size=(650, 650)):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        

        if os.path.isfile(input_path):
            try:
   
                output_path = os.path.join(output_folder, filename)
                

                resize_image(input_path, output_path, size)
            except Exception as e:
                print(f"Skipping {filename}: {e}")

if __name__ == "__main__":

    input_folder_path = '/Users/rajkhera/Downloads/test_set/test7'
    output_folder_path = '/Users/rajkhera/Downloads/test_set/test7_resized'


    resize_images_in_folder(input_folder_path, output_folder_path)
