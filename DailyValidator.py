import os
import tensorflow as tf
import GeoScript


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(base_dir, "Checkpoints")
    daily_images_dir = os.path.join(base_dir, "Daily Images")

    models = {}
    if not os.path.exists(checkpoints_dir):
        print(f"Directory not found: {checkpoints_dir}")
        return

    print("Loading models into memory...")
    for file in os.listdir(checkpoints_dir):
        if file.endswith(".keras"):
            model_path = os.path.join(checkpoints_dir, file)
            print(f"-> Loading {file}")
            models[file] = tf.keras.models.load_model(model_path)

    if not models:
        print("No .keras files found inside Checkpoints folder.")
        return

    if not os.path.exists(daily_images_dir):
        print(f"Directory not found: {daily_images_dir}")
        return

    for date_folder in os.listdir(daily_images_dir):
        date_path = os.path.join(daily_images_dir, date_folder)

        if os.path.isdir(date_path):
            print(f"\nEvaluating images inside: {date_folder}")

            images = [f for f in os.listdir(date_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                print(f"   No image files found in {date_folder}. Skipping...")
                continue

            output_txt_path = os.path.join(date_path, "predictions_summary.txt")

            with open(output_txt_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(f"MODEL PREDICTIONS FOR FOLDER: {date_folder}\n")
                txt_file.write("=" * 45 + "\n\n")

                for img_name in images:
                    img_path = os.path.join(date_path, img_name)
                    txt_file.write(f"Target Image File: {img_name}\n")
                    txt_file.write("-" * 35 + "\n")

                    for model_name, model_obj in models.items():
                        txt_file.write(f"[{model_name}]\n")

                        results = GeoScript.predict_image(model_obj, img_path)
                        for line in results:
                            txt_file.write(f"  {line}\n")
                        txt_file.write("\n")

                    txt_file.write("=" * 45 + "\n\n")

            print(f"-> Successfully written: {output_txt_path}")


if __name__ == "__main__":
    main()