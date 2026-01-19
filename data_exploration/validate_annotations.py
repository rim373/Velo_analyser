"""
Validate Annotations
Check that all masks are properly created and match their corresponding images.
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm


def validate_annotations(img_dir, mask_dir, output_dir=None, visualize=True):
    """
    Validate annotation quality and create visualization overlays.

    Args:
        img_dir: Directory with images
        mask_dir: Directory with masks
        output_dir: Directory to save validation visualizations
        visualize: Whether to create overlay visualizations
    """
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)

    if output_dir and visualize:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("VALIDATING ANNOTATIONS")
    print("="*60)
    print(f"Images: {img_dir}")
    print(f"Masks: {mask_dir}")
    print()

    # Track issues
    missing_masks = []
    size_mismatches = []
    non_binary_masks = []
    large_defects = []
    small_defects = []
    valid_count = 0

    # Get all image files
    image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

    print(f"Found {len(image_files)} images to validate...")
    print()

    for img_path in tqdm(image_files):
        # Look for mask
        mask_path = mask_dir / f"{img_path.stem}_mask.png"

        if not mask_path.exists():
            # Try without _mask suffix
            mask_path = mask_dir / f"{img_path.stem}.png"

        if not mask_path.exists():
            missing_masks.append(img_path.name)
            continue

        try:
            # Load image and mask
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), 0)

            if img is None:
                print(f"Error loading image: {img_path.name}")
                continue

            if mask is None:
                print(f"Error loading mask: {mask_path.name}")
                continue

            # Check 1: Size match
            if img.shape[:2] != mask.shape[:2]:
                size_mismatches.append({
                    'name': img_path.name,
                    'img_size': img.shape[:2],
                    'mask_size': mask.shape[:2]
                })
                continue

            # Check 2: Binary mask
            unique_values = np.unique(mask)
            if not (len(unique_values) <= 2 and all(v in [0, 255] for v in unique_values)):
                non_binary_masks.append({
                    'name': img_path.name,
                    'values': unique_values.tolist()
                })

            # Check 3: Defect coverage
            defect_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            defect_ratio = defect_pixels / total_pixels

            if defect_ratio > 0.5:
                large_defects.append({
                    'name': img_path.name,
                    'ratio': defect_ratio
                })
            elif defect_ratio < 0.01:
                small_defects.append({
                    'name': img_path.name,
                    'ratio': defect_ratio
                })

            # Create visualization
            if visualize and output_dir:
                # Create overlay
                overlay = img.copy()
                overlay[mask > 0] = [0, 0, 255]  # Red for defects
                blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

                # Create side-by-side comparison
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                comparison = np.hstack([img, mask_3ch, blended])

                # Add text
                h, w = comparison.shape[:2]
                cv2.putText(comparison, f"Original", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(comparison, f"Mask", (w//3 + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(comparison, f"Overlay ({defect_ratio:.1%})", (2*w//3 + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Save
                output_path = output_dir / f"{img_path.stem}_validation.jpg"
                cv2.imwrite(str(output_path), comparison)

            valid_count += 1

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    # Print report
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    print(f"Total images: {len(image_files)}")
    print(f"Valid annotations: {valid_count}")
    print()

    # Issues
    total_issues = len(missing_masks) + len(size_mismatches) + len(non_binary_masks)

    if total_issues == 0:
        print("[OK] All annotations are valid!")
    else:
        print(f"[WARNING] Found {total_issues} issues:")

    if missing_masks:
        print(f"\n1. Missing masks ({len(missing_masks)}):")
        for name in missing_masks[:5]:
            print(f"   - {name}")
        if len(missing_masks) > 5:
            print(f"   ... and {len(missing_masks) - 5} more")

    if size_mismatches:
        print(f"\n2. Size mismatches ({len(size_mismatches)}):")
        for item in size_mismatches[:5]:
            print(f"   - {item['name']}: img{item['img_size']} vs mask{item['mask_size']}")
        if len(size_mismatches) > 5:
            print(f"   ... and {len(size_mismatches) - 5} more")

    if non_binary_masks:
        print(f"\n3. Non-binary masks ({len(non_binary_masks)}):")
        for item in non_binary_masks[:5]:
            print(f"   - {item['name']}: values = {item['values']}")
        if len(non_binary_masks) > 5:
            print(f"   ... and {len(non_binary_masks) - 5} more")

    # Warnings
    if large_defects:
        print(f"\n[WARNING] Large defect areas ({len(large_defects)}):")
        print("(Defect covers >50% of image - verify this is correct)")
        for item in large_defects[:5]:
            print(f"   - {item['name']}: {item['ratio']:.1%}")
        if len(large_defects) > 5:
            print(f"   ... and {len(large_defects) - 5} more")

    if small_defects:
        print(f"\n[INFO] Small defect areas ({len(small_defects)}):")
        print("(Defect covers <1% of image)")
        for item in small_defects[:5]:
            print(f"   - {item['name']}: {item['ratio']:.1%}")
        if len(small_defects) > 5:
            print(f"   ... and {len(small_defects) - 5} more")

    if visualize and output_dir:
        print(f"\n[OK] Validation visualizations saved to: {output_dir}")

    # Save report
    report_path = Path("validation_report.txt")
    with open(report_path, 'w') as f:
        f.write("ANNOTATION VALIDATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total images: {len(image_files)}\n")
        f.write(f"Valid annotations: {valid_count}\n")
        f.write(f"Issues found: {total_issues}\n\n")

        if missing_masks:
            f.write(f"Missing masks ({len(missing_masks)}):\n")
            for name in missing_masks:
                f.write(f"  - {name}\n")
            f.write("\n")

        if size_mismatches:
            f.write(f"Size mismatches ({len(size_mismatches)}):\n")
            for item in size_mismatches:
                f.write(f"  - {item['name']}: img{item['img_size']} vs mask{item['mask_size']}\n")
            f.write("\n")

        if non_binary_masks:
            f.write(f"Non-binary masks ({len(non_binary_masks)}):\n")
            for item in non_binary_masks:
                f.write(f"  - {item['name']}: values = {item['values']}\n")
            f.write("\n")

        if large_defects:
            f.write(f"Large defects (>50% coverage) ({len(large_defects)}):\n")
            for item in large_defects:
                f.write(f"  - {item['name']}: {item['ratio']:.1%}\n")
            f.write("\n")

        if small_defects:
            f.write(f"Small defects (<1% coverage) ({len(small_defects)}):\n")
            for item in small_defects:
                f.write(f"  - {item['name']}: {item['ratio']:.1%}\n")

    print(f"\n[OK] Report saved to: {report_path}")

    return {
        'valid': valid_count,
        'missing': len(missing_masks),
        'size_mismatch': len(size_mismatches),
        'non_binary': len(non_binary_masks),
        'large_defects': len(large_defects),
        'small_defects': len(small_defects)
    }


def main():
    parser = argparse.ArgumentParser(description='Validate annotation quality')
    parser.add_argument('--img_dir', type=str,
                       default='data/annotated/damaged/images',
                       help='Directory with images')
    parser.add_argument('--mask_dir', type=str,
                       default='data/annotated/damaged/masks',
                       help='Directory with masks')
    parser.add_argument('--output_dir', type=str,
                       default='validation_output',
                       help='Directory to save validation visualizations')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Skip creating visualization overlays')

    args = parser.parse_args()

    results = validate_annotations(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        visualize=not args.no_visualize
    )

    # Exit with error code if issues found
    total_issues = results['missing'] + results['size_mismatch'] + results['non_binary']
    if total_issues > 0:
        print(f"\n[ERROR] Found {total_issues} issues that need fixing!")
        exit(1)
    else:
        print("\n[OK] All validations passed!")
        exit(0)


if __name__ == '__main__':
    main()
