import argparse
import cv2
from src.preprocess import MammogramPreprocess
from src.utils import load_chaincode, paint_abnormality
from src.visualize import visualize_results, visualize_steps

def main():
    parser = argparse.ArgumentParser(description="Mammogram Preprocessing Pipeline")
    parser.add_argument('--image_path', type=str, required=True, help='Path to mammogram image (.png)')
    parser.add_argument('--chaincode_path', type=str, required=True, help='Path to chaincode overlay file')
    parser.add_argument('--flag', type=int, default=2, choices=[1, 2], help='CLAHE flag: 1 (single), 2 (double)')
    parser.add_argument('--width', type=int, default=256, help='Target width')
    parser.add_argument('--height', type=int, default=512, help='Target height')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    args = parser.parse_args()

    # Load image and chaincode
    try:
        org_image = cv2.imread(args.image_path, 0)
        if org_image is None:
            raise ValueError(f"Failed to load image from {args.image_path}")
        chaincode = load_chaincode(args.chaincode_path)
    except Exception as e:
        print(f"Error loading inputs: {e}")
        return

    # Preprocess
    preprocessor = MammogramPreprocess(org_image, chaincode, args.flag, args.width, args.height)
    enhanced_image, bounding_box = preprocessor.preprocess()
    painted_image = preprocessor.plot_boundingbox(bounding_box)
    painted_org_image = paint_abnormality(org_image, chaincode)

    # Save enhanced image
    output_image_path = f"{args.output_dir}/enhanced_image_{args.image_path.split('/')[-1]}"
    cv2.imwrite(output_image_path, enhanced_image)
    print(f"Enhanced image saved to {output_image_path}")

    # Visualize
    visualize_results(org_image, painted_org_image, enhanced_image, painted_image, args.output_dir + "/plots")
    visualize_steps(preprocessor, args.output_dir + "/plots")

if __name__ == "__main__":
    main()
